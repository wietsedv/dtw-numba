import numpy as np
import pytest
from dtw_numba import dtw


def test_1d_2d():
    x1 = np.array([1.0, 2.0, 3.0])
    y1 = np.array([1.0, 2.0, 3.0, 4.0])
    d1 = dtw(x1, y1)

    x2 = np.array([[1.0], [2.0], [3.0]])
    y2 = np.array([[1.0], [2.0], [3.0], [4.0]])
    d2 = dtw(x2, y2)

    assert d1 == d2


def test_f32():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert type(dtw(x, y)) == float


def test_f16():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    with pytest.raises(TypeError):
        dtw(x, y)


def test_mixed():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    with pytest.raises(TypeError):
        dtw(x, y)
