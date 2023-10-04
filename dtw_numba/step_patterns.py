from typing import Callable, TypeAlias

import numba as nb
import numpy as np

import dtw_numba.windowing_functions as wf

StepPattern: TypeAlias = Callable[[np.ndarray, int, int, int], None]


@nb.njit(fastmath=True)
def symmetric1(
    matrix: np.ndarray,
    i: int,
    j: int,
    windowing_function: wf.WindowingFunction,
):
    def cost(i_, j_):
        return matrix[i_, j_] if windowing_function.check(i_, j_) else np.inf

    matrix[i, j] += min(cost(i, j - 1), cost(i - 1, j - 1), cost(i - 1, j))


@nb.njit(fastmath=True)
def symmetric2(
    matrix: np.ndarray,
    i: int,
    j: int,
    windowing_function: wf.WindowingFunction,
):
    def cost(i_, j_):
        return matrix[i_, j_] if windowing_function.check(i_, j_) else np.inf

    matrix[i, j] += min(cost(i, j - 1), 2 * cost(i - 1, j - 1), cost(i - 1, j))


@nb.njit(fastmath=True)
def asymmetric(
    matrix: np.ndarray,
    i: int,
    j: int,
    windowing_function: wf.WindowingFunction,
):
    def cost(i_, j_):
        return matrix[i_, j_] if windowing_function.check(i_, j_) else np.inf

    matrix[i, j] += min(cost(i, j - 1), cost(i - 1, j - 1), cost(i - 2, j - 1))
