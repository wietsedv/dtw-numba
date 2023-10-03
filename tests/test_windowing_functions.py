import numpy as np
import pytest

import dtw_numba.windowing_functions as wf


def _build_window(window_fn, args, n, m):
    actual = []
    for i in range(n):
        actual.append([])
        for j in range(m):
            actual[-1].append(window_fn(i, j, n, m, *args))
    actual_str = "\n".join([" ".join(["x" if x else "." for x in row]) for row in actual])
    return actual_str
    


def test_no_window():
    window_fn, *args = wf.no_window()
    actual = _build_window(window_fn, args, 8, 12)
    expected = (
        "x x x x x x x x x x x x\n"
        "x x x x x x x x x x x x\n"
        "x x x x x x x x x x x x\n"
        "x x x x x x x x x x x x\n"
        "x x x x x x x x x x x x\n"
        "x x x x x x x x x x x x\n"
        "x x x x x x x x x x x x\n"
        "x x x x x x x x x x x x"
    )
    assert actual == expected


def test_sakoe_chiba():
    window_fn, *args = wf.sakoe_chiba(2)
    actual = _build_window(window_fn, args, 8, 12)
    expected = (
        "x x x . . . . . . . . .\n"
        "x x x x . . . . . . . .\n"
        "x x x x x . . . . . . .\n"
        ". x x x x x . . . . . .\n"
        ". . x x x x x . . . . .\n"
        ". . . x x x x x . . . .\n"
        ". . . . x x x x x . . .\n"
        ". . . . . x x x x x . ."
    )
    assert actual == expected

# def test_itakura():
#     window_fn, *args = wf.itakura(2.0)
#     actual = _build_window(window_fn, args, 8, 12)
#     expected = (
#         ". x . . . . . . . . . .\n"
#         ". x x x . . . . . . . .\n"
#         ". x x x x . . . . . . .\n"
#         ". x x x x x . . . . . .\n"
#         ". . x x x x x . . . . .\n"
#         ". . . x x x x x . . . .\n"
#         ". . . . x x x x x . . .\n"
#         ". . . . . x x x x x . ."
#     )
#     assert actual == expected
