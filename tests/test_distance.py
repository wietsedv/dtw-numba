import numpy as np
import pytest
from dtw_numba import dtw


def test_dtype():
    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    assert dtw(x, y).dtype == np.float32


def test_distance():
    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([[0.5], [2.0], [3.5], [4.0]], dtype=np.float32)
    assert dtw(x, y) == pytest.approx(2.5)
