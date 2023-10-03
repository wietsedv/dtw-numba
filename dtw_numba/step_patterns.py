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
    windowing_args: tuple = (),
):
    matrix[i, j] += min(
        matrix[i, j - 1] if windowing_function(i, j - 1, *windowing_args) else np.inf,
        matrix[i - 1, j - 1]
        if windowing_function(i - 1, j - 1, *windowing_args)
        else np.inf,
        matrix[i - 1, j] if windowing_function(i - 1, j, *windowing_args) else np.inf,
    )


@nb.njit(fastmath=True)
def symmetric2(
    matrix: np.ndarray,
    i: int,
    j: int,
    windowing_function: wf.WindowingFunction,
    windowing_args: tuple = (),
):
    matrix[i, j] += min(
        matrix[i, j - 1] if windowing_function(i, j - 1, *windowing_args) else np.inf,
        (matrix[i - 1, j - 1] + matrix[i - 1, j - 1])
        if windowing_function(i - 1, j - 1, *windowing_args)
        else np.inf,
        matrix[i - 1, j] if windowing_function(i - 1, j, *windowing_args) else np.inf,
    )


@nb.njit(fastmath=True)
def asymmetric(
    matrix: np.ndarray,
    i: int,
    j: int,
    windowing_function: wf.WindowingFunction,
    windowing_args: tuple = (),
):
    matrix[i, j] += min(
        matrix[i, j - 1] if windowing_function(i, j - 1, *windowing_args) else np.inf,
        matrix[i - 1, j - 1]
        if windowing_function(i - 1, j - 1, *windowing_args)
        else np.inf,
        matrix[i - 2, j - 1]
        if windowing_function(i - 2, j - 1, *windowing_args)
        else np.inf,
    )
