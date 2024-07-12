.. dnn-inference documentation master file, created by
   sphinx-quickstart on Sun Aug  8 20:28:09 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

🦌 ReHLine
==========

.. raw:: html

    <embed>
        <a href="https://github.com/softmin/ReHLine"><img src="https://github.com/softmin/ReHLine-python/blob/main/doc/source/logo.png" align="right" height="138" /></a>
    </embed>

.. -*- mode: rst -*-

|PyPi|_ |MIT|_ |Python3|_ 

.. |PyPi| image:: https://badge.fury.io/py/rehline.svg
.. _PyPi: https://pypi.org/project/rehline/

.. |MIT| image:: https://img.shields.io/pypi/l/dnn-inference.svg
.. _MIT: https://opensource.org/licenses/MIT

.. |Python3| image:: https://img.shields.io/badge/python-3-blue.svg
.. _Python3: www.python.org

.. |downloads| image:: https://pepy.tech/badge/rehline
.. _downloads: https://pepy.tech/project/rehline


**ReHLine** is designed to be a computationally efficient and practically useful software package for large-scale ERMs. 

- GitHub repo: `https://github.com/softmin/ReHLine-python <https://github.com/softmin/ReHLine-python>`_
- Documentation: `https://rehline.readthedocs.io <https://rehline.readthedocs.io/en/latest/>`_
- PyPi: `https://pypi.org/project/rehline <https://pypi.org/project/rehline>`_
- Open Source: `MIT license <https://opensource.org/licenses/MIT>`_
- Paper: `NeurIPS | 2023 <https://openreview.net/pdf?id=3pEBW2UPAD>`_


`ReHLine` is designed to be a computationally efficient 
and practically useful software package for large-scale ERMs.

The proposed **ReHLine** solver has four appealing `linear properties`:

- It applies to any convex piecewise linear-quadratic loss function, including the hinge loss, the check loss, the Huber loss, etc.
- In addition, it supports linear equality and inequality constraints on the parameter vector.
- The optimization algorithm has a provable linear convergence rate.
- The per-iteration computational complexity is linear in the sample size.

🔨 Installation
===============

Install ``rehline`` using ``pip``

.. code:: bash

	pip install rehline

See more details in `installation <./installation.rst>`_.

📮 Formulation
--------------

`ReHLine` is designed to address the empirical regularized ReLU-ReHU minimization problem, named *ReHLine optimization*, of the following form:

.. math::

  \min_{\mathbf{\beta} \in \mathbb{R}^d} \sum_{i=1}^n \sum_{l=1}^L \text{ReLU}( u_{li} \mathbf{x}_i^\intercal \mathbf{\beta} + v_{li}) + \sum_{i=1}^n \sum_{h=1}^H {\text{ReHU}}_{\tau_{hi}}( s_{hi} \mathbf{x}_i^\intercal \mathbf{\beta} + t_{hi}) + \frac{1}{2} \| \mathbf{\beta} \|_2^2, \ \text{ s.t. } \mathbf{A} \mathbf{\beta} + \mathbf{b} \geq \mathbf{0},


where :math:`\mathbf{U} = (u_{li}),\mathbf{V} = (v_{li}) \in \mathbb{R}^{L \times n}` 
and :math:`\mathbf{S} = (s_{hi}),\mathbf{T} = (t_{hi}),\mathbf{\tau} = (\tau_{hi}) \in \mathbb{R}^{H \times n}` 
are the ReLU-ReHU loss parameters, and :math:`(\mathbf{A},\mathbf{b})` are the constraint parameters. 
This formulation has a wide range of applications spanning various fields, including statistics, 
machine learning, computational biology, and social studies. 
Some popular examples include SVMs with fairness constraints (FairSVM), 
elastic net regularized quantile regression (ElasticQR), 
and ridge regularized Huber minimization (RidgeHuber).

.. image:: ./figs/tab.png

📚 Benchmark (powered by `benchopt`)
------------------------------------

To generate benchmark results in our paper, please check `ReHLine-benchmark <https://github.com/softmin/ReHLine-benchmark>`_.

+-------------+--------------------------------------------------------+
| Problem     | Results                                                |
+=============+========================================================+
| FairSVM_    | `Result <./_static/benchmark/benchmark_FairSVM.html>`_ |
+-------------+--------------------------------------------------------+
| ElasticQR_  | `Result <./_static/benchmark/benchmark_QR.html>`_      |
+-------------+--------------------------------------------------------+
| RidgeHuber_ | `Result <./_static/benchmark/benchmark_Huber.html>`_   |
+-------------+--------------------------------------------------------+
| SVM_        | `Result <./_static/benchmark/benchmark_SVM.html>`_     |
+-------------+--------------------------------------------------------+
| sSVM_       | `Result <./_static/benchmark/benchmark_sSVM.html>`_    |
+-------------+--------------------------------------------------------+

.. _FairSVM: https://github.com/softmin/ReHLine-benchmark/tree/main/benchmark_FairSVM
.. _ElasticQR: https://github.com/softmin/ReHLine-benchmark/tree/main/benchmark_QR
.. _RidgeHuber: https://github.com/softmin/ReHLine-benchmark/tree/main/benchmark_Huber
.. _SVM: https://github.com/softmin/ReHLine-benchmark/tree/main/benchmark_SVM
.. _sSVM: https://github.com/softmin/ReHLine-benchmark/tree/main/benchmark_sSVM

*Note*: You may select the "log-log scale" option in the left sidebar, as this will significantly improve the readability of the results.

🧾 Overview of Results
----------------------

.. image:: ./figs/res.png

Reference
---------
**If you use this code please star the repository and cite the following paper:**

.. code:: bib

   @article{daiqiu2023rehline,
   title={ReHLine: Regularized Composite ReLU-ReHU Loss Minimization with Linear Computation and Linear Convergence},
   author={Dai, Ben and Yixuan Qiu},
   journal={Advances in Neural Information Processing Systems},
   year={2023},
   }

📒 Contents
-----------

.. toctree::
   :maxdepth: 2

   installation
   example

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
   
.. quickstart
