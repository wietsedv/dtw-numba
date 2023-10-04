from typing import Callable, TypeAlias

import numba as nb
import numpy as np

import dtw_numba.windowing_functions as wf


DistanceMetric: TypeAlias = Callable[
    [np.ndarray, np.ndarray, wf.WindowingFunction], np.ndarray
]


@nb.njit(parallel=True, fastmath=True)
def euclidean(
    x: np.ndarray,
    y: np.ndarray,
    windowing_function: wf.WindowingFunction
):
    matrix = np.empty((x.shape[0], y.shape[0]), dtype=x.dtype)
    for i in nb.prange(x.shape[0]):
        for j in nb.prange(y.shape[0]):
            if windowing_function.check(i, j) is False:
                continue
            matrix[i, j] = 0.0
            for k in nb.prange(x.shape[1]):
                matrix[i, j] += (x[i, k] - y[j, k]) ** 2
            matrix[i, j] = np.sqrt(matrix[i, j])
    return matrix
