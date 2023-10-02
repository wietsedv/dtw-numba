from enum import IntEnum

from numba import njit
import numpy as np


class StepPattern(IntEnum):
    symmetric1 = 1
    symmetric2 = 2
    asymmetric = 3


@njit(fastmath=True)
def symmetric1(matrix: np.ndarray, i: int, j: int, w: int):
    matrix[i, j] += min(
        matrix[i, j - 1] if j > 0 and (w == 0 or (i - j) < w) else np.inf,
        matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf,
        matrix[i - 1, j] if i > 0 and (w == 0 or (i - j) > -w) else np.inf,
    )


@njit(fastmath=True)
def symmetric2(matrix: np.ndarray, i: int, j: int, w: int):
    matrix[i, j] += min(
        matrix[i, j - 1] if j > 0 and (w == 0 or (i - j) < w) else np.inf,
        matrix[i - 1, j - 1] + matrix[i, j] if i > 0 and j > 0 else np.inf,
        matrix[i - 1, j] if i > 0 and (w == 0 or (i - j) > -w) else np.inf,
    )


@njit(fastmath=True)
def asymmetric(matrix: np.ndarray, i: int, j: int, w: int):
    matrix[i, j] += min(
        matrix[i, j - 1] if j > 0 and (w == 0 or (i - j) < w) else np.inf,
        matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf,
        matrix[i - 2, j - 1] if i > 1 and (w == 0 or (i - j) > -w) else np.inf,
    )
