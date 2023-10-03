from typing import Callable, TypeAlias

import numba as nb

WindowingFunction: TypeAlias = Callable[[int, int, int, int], bool]


@nb.njit(fastmath=True)
def _no_window(i: int, j: int, n: int, m: int):
    return i >= 0 and j >= 0


def no_window():
    return (_no_window,)


@nb.njit(fastmath=True)
def _sakoe_chiba(i: int, j: int, n: int, m: int, window_size: int):
    if window_size < 0 or i < 0 or j < 0:
        return False
    return abs(i - j) <= window_size


def sakoe_chiba(window_size: int):
    """Sakoe-Chiba band windowing function."""
    return _sakoe_chiba, window_size


@nb.njit(fastmath=True)
def _slanted_band(i: int, j: int, n: int, m: int, window_size: int):
    if window_size < 0 or i < 0 or j < 0:
        return False
    return abs(j - i * m / n) <= window_size


def slanted_band(window_size: int):
    """Slanted band windowing function."""
    return _slanted_band, window_size
