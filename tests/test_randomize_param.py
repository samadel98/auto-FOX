"""Tests for :mod:`FOX.functions.randomize_param`."""

import numpy as np

from FOX.functions.randomize_param import randomize


def test_randomize_default_bounds() -> None:
    np.random.seed(1)
    val = np.array([1.0, 2.0, 4.0])
    ret = randomize(val)
    assert ret.shape == val.shape
    assert (ret >= val / 2).all() and (ret <= val * 2).all()


def test_randomize_bounds() -> None:
    np.random.seed(1)
    val = np.array([1.0, 2.0, 4.0])
    val_max = np.array([1.5, np.inf, 5])
    val_min = np.array([0.9, 1.9, -np.inf])
    for _ in range(100):
        ret = randomize(val, val_max=val_max, val_min=val_min)
        assert (ret >= [0.9, 1.9, 2.0]).all(), ret
        assert (ret <= [1.5, 4.0, 5.0]).all(), ret


def test_randomize_scalar_bounds() -> None:
    np.random.seed(1)
    val = np.array([1.0, 2.0, 4.0])
    for _ in range(100):
        ret = randomize(val, val_max=3, val_min=1.5)
        assert (ret >= [1.5, 1.5, 2.0]).all(), ret
        assert (ret <= [2.0, 3.0, 3.0]).all(), ret
