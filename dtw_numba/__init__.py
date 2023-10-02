import sys

import numpy as np

from distance_metrics import euclidean
from dynamic_time_warping import dynamic_time_warping
from step_patterns import StepPattern


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

    matrix = euclidean(x, y, window_size)
    return dynamic_time_warping(matrix, window_size, step_pattern)


if __name__ == "__main__":
    import time

    np.random.seed(1)
    x = np.random.random((1500, 1024))
    y = np.random.random((2000, 1024))
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    t0 = time.time()
    for _ in range(20):
        d = dtw(x, y, window_size=500)
    t1 = time.time()
    print(f"t={(t1 - t0) / 20 * 1000:.1f}ms ({d / (x.shape[0] + y.shape[0]):.3f})")
