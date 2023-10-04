import numba as nb


class WindowingFunction:
    m: int

    def __init__(self):
        pass

    def set_bounds(self, n: int, m: int):
        self.m = m

    def check(self, i: int, j: int):
        return i >= 0 and j >= 0

    def range(self, i: int):
        return 0, self.m


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

    def range(self, i: int):
        return max(0, i - self.window_size), min(self.m, i + self.window_size + 1)


@nb.experimental.jitclass
class SlantedBand(WindowingFunction):
    m: int
    slope: float
    window_size: int

    def __init__(self, window_size: int):
        self.window_size = window_size

    def set_bounds(self, n: int, m: int):
        self.m = m
        self.slope = m / n

    def check(self, i: int, j: int):
        if i < 0 or j < 0:
            return False
        return abs(i * self.slope - j) <= self.window_size

    def range(self, i: int):
        i_s = i * self.slope
        return max(0, i_s - self.window_size), min(self.m, i_s + self.window_size + 1)
