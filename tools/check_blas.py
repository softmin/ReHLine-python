"""Check BLAS matrix products against direct summation before numerical tests."""

import json

import numpy as np
from threadpoolctl import threadpool_info


def main():
    print(json.dumps({"numpy": np.__version__, "libraries": threadpool_info()}, indent=2))
    rng = np.random.default_rng(917)
    # Skinny and Gram products exercise the legacy OpenBLAS failure reported in
    # https://github.com/numpy/numpy/issues/24903 without involving ReHLine.
    X = rng.normal(size=(2000, 10))
    products = [(rng.normal(size=(625, 5)), rng.normal(size=(5, 625))), (X.T, X)]
    for left, right in products:
        reference = np.einsum("ik,kj->ij", left, right, optimize=False)
        for actual in (left @ right, (right.T @ left.T).T):
            np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
    print("BLAS matrix products agree with independent summation.")


if __name__ == "__main__":
    main()
