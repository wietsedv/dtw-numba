from enum import IntEnum

from numba import njit, prange
import numpy as np


class DistanceMetric(IntEnum):
    euclidean = 1


@njit(parallel=True, fastmath=True)
def _euclidean(x: np.ndarray, y: np.ndarray, w: int):
    matrix = np.empty((x.shape[0], y.shape[0]), dtype=x.dtype)
    for i in prange(x.shape[0]):
        if w == 0:
            j_start, j_stop = 0, y.shape[0]
        else:
            j_start = max(0, i - w)
            j_stop = min(y.shape[0], i + w + 1)
        for j in prange(j_start, j_stop):
            matrix[i, j] = 0.0
            for k in prange(x.shape[1]):
                matrix[i, j] += (x[i, k] - y[j, k]) ** 2
            matrix[i, j] = np.sqrt(matrix[i, j])
    return matrix


@njit("float32[:,::1](float32[:,::1],float32[:,::1],int64)")
def euclidean_f32(x: np.ndarray, y: np.ndarray, w: int):
    return _euclidean(x, y, w)


@njit("float64[:,::1](float64[:,::1],float64[:,::1],int64)")
def euclidean_f64(x: np.ndarray, y: np.ndarray, w: int):
    return _euclidean(x, y, w)


def euclidean(x: np.ndarray, y: np.ndarray, w: int):
    if x.dtype == np.float32:
        return euclidean_f32(x, y, w)
    elif x.dtype == np.float64:
        return euclidean_f64(x, y, w)
    raise TypeError("x must have a float32 or float64 dtype")
