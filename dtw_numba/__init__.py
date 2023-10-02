import sys

import numba as nb
import numpy as np


@nb.njit(
    "float32[:,::1](float32[:,::1],float32[:,::1],int64)", parallel=True, fastmath=True
)
def _euclidean(x: np.ndarray, y: np.ndarray, w: int):
    n = x.shape[0]
    m = y.shape[0]
    matrix = np.zeros((n, m), dtype=x.dtype)

    for i in nb.prange(n):
        for j in nb.prange(0 if w == 0 else max(0, i - w), m if w == 0 else min(m, i + w + 1)):
            for k in nb.prange(x.shape[1]):
                matrix[i, j] += (x[i, k] - y[j, k]) ** 2
            matrix[i, j] = np.sqrt(matrix[i, j])
    return matrix


@nb.njit("void(float32[:,::1],int64)", fastmath=True)
def _dynamic_time_warping(matrix: np.ndarray, w: int | None):
    n, m = matrix.shape
    for i in range(1, n):
        matrix[i, 0] += matrix[i - 1, 0]
    for j in range(1, m):
        matrix[0, j] += matrix[0, j - 1]
    for i in range(1, n):
        for j in range(1, m):
            if w == 0:
                matrix[i, j] += min(
                    matrix[i - 1, j - 1] + matrix[i, j],
                    matrix[i, j - 1],
                    matrix[i - 1, j],
                )
                continue

            offset = i - j
            if (offset < -w) or (offset > w):
                continue

            if offset == -w:
                matrix[i, j] += min(
                    matrix[i - 1, j - 1] + matrix[i, j],
                    matrix[i, j - 1],
                )
            elif offset == w:
                matrix[i, j] += min(
                    matrix[i - 1, j - 1] + matrix[i, j],
                    matrix[i - 1, j],
                )
            else:
                matrix[i, j] += min(
                    matrix[i - 1, j - 1] + matrix[i, j],
                    matrix[i, j - 1],
                    matrix[i - 1, j],
                )


def dtw(
    x: np.ndarray,
    y: np.ndarray,
    *,
    # TODO distance_metric: str = "euclidean",
    # TODO window_type: str = "sakoechiba",
    window_size: int = 0,
    # TODO step_pattern="symmetric2",
):
    if len(x.shape) != 2 or len(y.shape) != 2:
        raise ValueError("the input arrays must both have two dimensions")
    if x.shape[1] != y.shape[1]:
        raise ValueError("the second dimensions of the input arrays must match in size")

    if window_size > 0:
        min_window_size = abs(x.shape[0] - y.shape[0])
        if window_size is not None and window_size < min_window_size:
            print(f"WARNING: window_size ({window_size}) is smaller than |x - y| ({min_window_size}). implicitly increased window size", file=sys.stderr)

    matrix = _euclidean(x, y, window_size)
    _dynamic_time_warping(matrix, window_size)

    return matrix[-1, -1]
