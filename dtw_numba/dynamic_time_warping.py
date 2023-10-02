from numba import njit
import numpy as np

from step_patterns import StepPattern, symmetric1, symmetric2, asymmetric


@njit(fastmath=True)
def _dynamic_time_warping(matrix: np.ndarray, w: int, s: StepPattern):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == 0 and j == 0:
                continue
            if w > 0 and abs(i - j) > w:
                continue
            if s == StepPattern.symmetric1:
                symmetric1(matrix, i, j, w)
            elif s == StepPattern.symmetric2:
                symmetric2(matrix, i, j, w)
            elif s == StepPattern.asymmetric:
                asymmetric(matrix, i, j, w)
            else:
                raise ValueError("invalid step pattern")

    return matrix[-1, -1]


@njit("float32(float32[:,::1],int64,int64)")
def _dynamic_time_warping_f32(matrix: np.ndarray, w: int, s: StepPattern):
    return _dynamic_time_warping(matrix, w, s)


@njit("float64(float64[:,::1],int64,int64)")
def _dynamic_time_warping_f64(matrix: np.ndarray, w: int, s: StepPattern):
    return _dynamic_time_warping(matrix, w, s)


def dynamic_time_warping(matrix: np.ndarray, w: int, s: StepPattern):
    if matrix.dtype == np.float32:
        return _dynamic_time_warping_f32(matrix, w, s)
    elif matrix.dtype == np.float64:
        return _dynamic_time_warping_f64(matrix, w, s)
    raise TypeError("matrix must have a float32 or float64 dtype")
