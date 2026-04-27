"""Type stubs for the rehline._internal C extension module."""

import numpy as np
import numpy.typing as npt

class rehline_result:
    beta: npt.NDArray[np.float64]
    xi: npt.NDArray[np.float64]
    Lambda: npt.NDArray[np.float64]
    Gamma: npt.NDArray[np.float64]
    mu: npt.NDArray[np.float64]
    niter: int
    dual_objfns: list[float]
    primal_objfns: list[float]
    def __init__(self) -> None: ...

def rehline_internal(
    result: rehline_result,
    X: npt.NDArray[np.float64],
    A: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    rho: npt.NDArray[np.float64],
    U: npt.NDArray[np.float64],
    V: npt.NDArray[np.float64],
    S: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
    Tau: npt.NDArray[np.float64],
    max_iter: int,
    tol: float,
    shrink: int = ...,
    verbose: int = ...,
    trace_freq: int = ...,
) -> None: ...
