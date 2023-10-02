import numpy as np

from distance_metrics import DistanceMetric, euclidean
from dynamic_time_warping import dynamic_time_warping
from step_patterns import StepPattern
from window_types import WindowType


def _sanitize_input(
    q: np.ndarray,
    r: np.ndarray,
    window_type: WindowType,
    window_size: int,
):
    if len(q.shape) == 1:
        q = q.reshape(-1, 1)
    if len(r.shape) == 1:
        r = r.reshape(-1, 1)

    # input
    if len(q.shape) != 2 or len(r.shape) != 2:
        raise ValueError("the input arrays must both have one or two dimensions")
    if q.shape[1] != r.shape[1]:
        raise ValueError("the second dimensions of the input arrays must match in size")
    if q.dtype != r.dtype:
        raise TypeError("the two input arrays must have the same dtype")
    if q.dtype != np.float32 and q.dtype != np.float64:
        raise TypeError("the input arrays must have a float32 or float64 dtype")

    # window
    if window_size > 0 and window_type == WindowType.sakoechiba:
        if window_size < abs(q.shape[0] - r.shape[0]):
            raise ValueError(
                "window size cannot be smaller than |x - y| when using the sakoechiba window"
            )

    return q, r, window_size


def dtw(
    q: np.ndarray,
    r: np.ndarray,
    *,
    distance_metric: DistanceMetric = DistanceMetric.euclidean,
    step_pattern: StepPattern = StepPattern.symmetric2,
    window_type: WindowType = WindowType.sakoechiba,
    window_size: int = 0,
):
    q, r, window_size = _sanitize_input(q, r, window_type, window_size)

    if distance_metric == DistanceMetric.euclidean:
        matrix = euclidean(q, r, window_size)
    else:
        raise ValueError("invalid distance metric")

    return dynamic_time_warping(matrix, window_size, step_pattern)


def main():
    import time

    np.random.seed(1)
    q = np.random.random((1500, 1024))
    r = np.random.random((2000, 1024))
    q = q.astype(np.float32, copy=False)
    r = r.astype(np.float32, copy=False)
    t0 = time.time()
    for _ in range(20):
        d = dtw(q, r, window_size=500)
    t1 = time.time()
    print(f"t={(t1 - t0) / 20 * 1000:.1f}ms ({d / (q.shape[0] + r.shape[0]):.3f})")


if __name__ == "__main__":
    main()
