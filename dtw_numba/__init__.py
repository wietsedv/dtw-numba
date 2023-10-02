from enum import IntEnum
import sys

import numba as nb
import numpy as np


class StepPattern(IntEnum):
    symmetric1 = 1
    symmetric2 = 2
    asymmetric = 3


@nb.njit(parallel=True, fastmath=True)
def _euclidean(x: np.ndarray, y: np.ndarray, w: int):
    matrix = np.empty((x.shape[0], y.shape[0]), dtype=x.dtype)
    for i in nb.prange(x.shape[0]):
        if w == 0:
            j_start, j_stop = 0, y.shape[0]
        else:
            j_start = max(0, i - w)
            j_stop = min(y.shape[0], i + w + 1)
        for j in nb.prange(j_start, j_stop):
            matrix[i, j] = 0.0
            for k in nb.prange(x.shape[1]):
                matrix[i, j] += (x[i, k] - y[j, k]) ** 2
            matrix[i, j] = np.sqrt(matrix[i, j])
    return matrix


@nb.njit("float32[:,::1](float32[:,::1],float32[:,::1],int64)")
def _euclidean_f32(x: np.ndarray, y: np.ndarray, w: int):
    return _euclidean(x, y, w)


@nb.njit("float64[:,::1](float64[:,::1],float64[:,::1],int64)")
def _euclidean_f64(x: np.ndarray, y: np.ndarray, w: int):
    return _euclidean(x, y, w)


@nb.njit(fastmath=True)
def _symmetric1(matrix: np.ndarray, i: int, j: int, w: int):
    matrix[i, j] += min(
        matrix[i, j - 1] if w == 0 or (i - j) < w else np.inf,
        matrix[i - 1, j - 1],
        matrix[i - 1, j] if w == 0 or (i - j) > -w else np.inf,
    )

@nb.njit(fastmath=True)
def _symmetric2(matrix: np.ndarray, i: int, j: int, w: int):
    matrix[i, j] += min(
        matrix[i, j - 1] if w == 0 or (i - j) < w else np.inf,
        matrix[i - 1, j - 1] + matrix[i, j],
        matrix[i - 1, j] if w == 0 or (i - j) > -w else np.inf,
    )

@nb.njit(fastmath=True)
def _asymmetric(matrix: np.ndarray, i: int, j: int, w: int):
    matrix[i, j] += min(
        matrix[i, j - 1] if w == 0 or (i - j) < w else np.inf,
        matrix[i - 1, j - 1],
        matrix[i - 2, j - 1] if i > 1 and (w == 0 or (i - j) > -w) else np.inf,
    )

@nb.njit(fastmath=True)
def _dynamic_time_warping(matrix: np.ndarray, w: int, s: StepPattern):
    n, m = matrix.shape

    # first column
    for i in range(1, n if w == 0 else min(n, w + 1)):
        matrix[i, 0] += matrix[i - 1, 0]

    # first row
    for j in range(1, m if w == 0 else min(m, w + 1)):
        matrix[0, j] += matrix[0, j - 1]

    # rest of the matrix
    for i in range(1, n):
        for j in range(1, m):
            if w > 0 and abs(i - j) > w:
                continue
            if s == StepPattern.symmetric1:
                _symmetric1(matrix, i, j, w)
            elif s == StepPattern.symmetric2:
                _symmetric2(matrix, i, j, w)
            elif s == StepPattern.asymmetric:
                _asymmetric(matrix, i, j, w)

    return matrix[-1, -1]


@nb.njit("float32(float32[:,::1],int64,int64)")
def _dynamic_time_warping_f32(matrix: np.ndarray, w: int, s: StepPattern):
    return _dynamic_time_warping(matrix, w, s)


@nb.njit("float64(float64[:,::1],int64,int64)")
def _dynamic_time_warping_f64(matrix: np.ndarray, w: int, s: StepPattern):
    return _dynamic_time_warping(matrix, w, s)


def _sanitize_input(x: np.ndarray, y: np.ndarray, window_size: int):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # input
    if len(x.shape) != 2 or len(y.shape) != 2:
        raise ValueError("the input arrays must both have one or two dimensions")
    if x.shape[1] != y.shape[1]:
        raise ValueError("the second dimensions of the input arrays must match in size")
    if x.dtype != y.dtype:
        raise TypeError("the two input arrays must have the same dtype")
    if x.dtype != np.float32 and x.dtype != np.float64:
        raise TypeError("the input arrays must have a float32 or float64 dtype")

    # window_size
    if window_size > 0:
        min_window_size = abs(x.shape[0] - y.shape[0])
        if window_size is not None and window_size < min_window_size:
            print(
                f"WARNING: window_size ({window_size}) is smaller than |x - y| ({min_window_size}). implicitly increased window size",
                file=sys.stderr,
            )

    return x, y, window_size


def dtw(
    x: np.ndarray,
    y: np.ndarray,
    *,
    # TODO distance_metric: str = "euclidean",
    # TODO window_type: str = "sakoechiba",
    window_size: int = 0,
    step_pattern: StepPattern = StepPattern.symmetric2,
):
    x, y, window_size = _sanitize_input(x, y, window_size)

    double_precision = x.dtype == np.float64

    metric_fn = _euclidean_f64 if double_precision else _euclidean_f32
    matrix = metric_fn(x, y, window_size)

    dtw_fn = (
        _dynamic_time_warping_f64 if double_precision else _dynamic_time_warping_f32
    )
    return dtw_fn(matrix, window_size, step_pattern)


if __name__ == "__main__":
    import time

    np.random.seed(1)
    x = np.random.random((1500, 1024))
    y = np.random.random((2000, 1024))
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    t0 = time.time()
    for _ in range(20):
        d = dtw(x, y, window_size=500, step_pattern=StepPattern.asymmetric)
    t1 = time.time()
    print(f"t={(t1 - t0) / 20 * 1000:.1f}ms ({d / (x.shape[0] + y.shape[0]):.3f})")
