import numba as nb
import numpy as np

import dtw_numba.distance_metrics as dm
import dtw_numba.step_patterns as sp
import dtw_numba.windowing_functions as wf


def _validate_input(
    q: np.ndarray,
    r: np.ndarray,
    windowing_function: wf.WindowingFunction,
):
    if len(q.shape) != 2 or len(r.shape) != 2:
        raise ValueError("the input arrays must both have one or two dimensions")
    if q.shape[1] != r.shape[1]:
        raise ValueError("the second dimensions of the input arrays must match in size")
    if q.dtype != r.dtype:
        raise TypeError("the two input arrays must have the same dtype")
    if q.dtype != np.float32 and q.dtype != np.float64:
        raise TypeError("the input arrays must have a float32 or float64 dtype")
    if windowing_function.check(q.shape[0] - 1, r.shape[0] - 1) is False:
        raise ValueError(
            "windowing function must return True for final row/column. increase window size"
        )


@nb.njit(fastmath=True)
def _dtw(
    q: np.ndarray,
    r: np.ndarray,
    distance_metric: dm.DistanceMetric,
    step_pattern: sp.StepPattern,
    windowing_function: wf.WindowingFunction,
):
    matrix = distance_metric(q, r, windowing_function)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if windowing_function.check(i, j) is False:
                continue
            if i > 0 or j > 0:
                step_pattern(matrix, i, j, windowing_function)
    return matrix[-1, -1]


def dtw(
    q: np.ndarray,
    r: np.ndarray,
    *,
    distance_metric: dm.DistanceMetric = dm.euclidean,
    step_pattern: sp.StepPattern = sp.symmetric2,
    windowing_function: wf.WindowingFunction | None = None,
):
    if len(q.shape) == 1:
        q = q.reshape(-1, 1)
    if len(r.shape) == 1:
        r = r.reshape(-1, 1)

    if windowing_function is None:
        windowing_function = wf.NoWindow()
    windowing_function.set_bounds(q.shape[0], r.shape[0])

    _validate_input(q, r, windowing_function)

    return _dtw(
        q,
        r,
        distance_metric,
        step_pattern,
        windowing_function,
    )


def main():
    import time

    np.random.seed(1)
    q = np.random.random((2048, 1024))
    r = np.random.random((2048, 1024))
    q = q.astype(np.float32, copy=False)
    r = r.astype(np.float32, copy=False)

    # windowing_function = wf.NoWindow()
    windowing_function = wf.SakoeChiba(500)
    # windowing_function = wf.SlantedBand(500)
    for _ in range(10):
        d = dtw(q, r, windowing_function=windowing_function)
    t0 = time.time()
    for _ in range(20):
        d = dtw(q, r, windowing_function=windowing_function)
    t1 = time.time()
    print(f"t={(t1 - t0) / 20 * 1000:.1f}ms ({d / (q.shape[0] + r.shape[0]):.3f})")

    # window_fn, *args = wf.slanted_band(4)
    # n, m = 24, 24
    # actual = np.empty((n, m), dtype=np.bool_)
    # for i in range(n):
    #     for j in range(m):
    #         actual[i, j] = window_fn(i, j, n, m, *args)
    # for row in actual:
    #     print(" ".join(["x" if x else "." for x in row]), str(sum(row)).zfill(2))


if __name__ == "__main__":
    main()
