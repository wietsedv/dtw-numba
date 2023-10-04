from typing import Any, Callable, TypeAlias

import numba as nb


class WindowingFunction:
    n: int
    m: int

    def __init__(self):
        self.n = 0
        self.m = 0

    def set_bounds(self, n: int, m: int):
        self.n = n
        self.m = m

    def check(self, i: int, j: int) -> Any:
        return i >= 0 and j >= 0


@nb.experimental.jitclass()
class NoWindow(WindowingFunction):
    pass


@nb.experimental.jitclass
class SakoeChiba(WindowingFunction):
    window_size: int

    def __init__(self, window_size: int):
        self.n = 0
        self.m = 0
        self.window_size = window_size

    def check(self, i: int, j: int) -> Any:
        return i >= 0 and j >= 0 and abs(i - j) <= self.window_size


@nb.experimental.jitclass
class SlantedBand(WindowingFunction):
    window_size: int

    def __init__(self, window_size: int):
        self.n = 0
        self.m = 0
        self.window_size = window_size

    def check(self, i: int, j: int) -> Any:
        return i >= 0 and j >= 0 and abs(j - i * self.m / self.n) <= self.window_size
