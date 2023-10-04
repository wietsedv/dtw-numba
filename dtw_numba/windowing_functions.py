import numba as nb


class WindowingFunction:
    def __init__(self):
        pass

    def set_bounds(self, n: int, m: int):
        pass

    def check(self, i: int, j: int):
        return i >= 0 and j >= 0


@nb.experimental.jitclass()
class NoWindow(WindowingFunction):
    pass


@nb.experimental.jitclass
class SakoeChiba(WindowingFunction):
    window_size: int

    def __init__(self, window_size: int):
        self.window_size = window_size

    def check(self, i: int, j: int):
        if i < 0 or j < 0:
            return False
        return abs(i - j) <= self.window_size


@nb.experimental.jitclass
class SlantedBand(WindowingFunction):
    window_size: int
    slope: float

    def __init__(self, window_size: int):
        self.window_size = window_size

    def set_bounds(self, n: int, m: int):
        self.slope = m / n

    def check(self, i: int, j: int):
        if i < 0 or j < 0:
            return False
        return abs(j - i * self.slope) <= self.window_size
