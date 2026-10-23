"""Tests for :mod:`FOX.armc.err_funcs`."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import numpy as np
import pandas as pd

from FOX.armc import (
    mse_normalized,
    mse_normalized_weighted,
    mse_normalized_max,
    mse_normalized_v2,
    mse_normalized_weighted_v2,
    err_normalized,
    err_normalized_weighted,
    default_error_func,
)

QM = np.array([[1.0, 2.0], [3.0, 4.0]])
MM = np.array([[1.0, 1.0], [2.0, 2.0]])

# Reference values computed by hand from `QM` and `MM`;
# the "weighted" functions treat each row as a separate (flattened) array
REF: dict[Callable[..., np.float64], float] = {
    mse_normalized: (0 + 1 + 1 + 4) / 10,
    mse_normalized_weighted: ((1 / 3)**2 + (5 / 7)**2) / 2,
    mse_normalized_max: 5 / 7,
    mse_normalized_v2: (0**2 + 1**2 + 1**2 + 2**2) / 10**2,
    mse_normalized_weighted_v2: ((1 / 9)**2 + (5 / 49)**2) / 2,
    err_normalized: (0 + 1 + 1 + 2) / 10,
    err_normalized_weighted: ((1 / 3)**2 + (3 / 7)**2) / 2,
}

WEIGHTED = [
    mse_normalized_weighted,
    mse_normalized_max,
    mse_normalized_weighted_v2,
    err_normalized_weighted,
]
ZERO_CHECKED = [func for func in REF if func is not mse_normalized]


def test_default_error_func() -> None:
    assert default_error_func is mse_normalized


@pytest.mark.parametrize("func,ref", REF.items(), ids=lambda i: getattr(i, "__name__", ""))
def test_err_func(func: Callable[..., np.float64], ref: float) -> None:
    """Test the error functions against hand-computed values."""
    out = func(QM, MM)
    np.testing.assert_allclose(out, ref)
    np.testing.assert_allclose(func(QM.tolist(), MM.tolist()), ref)

    # The error is 0 for identical input
    np.testing.assert_allclose(func(QM, QM), 0.0)


@pytest.mark.parametrize("func", WEIGHTED, ids=lambda i: i.__name__)
def test_err_func_dataframe(func: Callable[..., np.float64]) -> None:
    """Test that a DataFrame's columns, rather than its rows, are treated as separate arrays."""
    ref = func(QM, MM)
    out = func(pd.DataFrame(QM.T), pd.DataFrame(MM.T))
    np.testing.assert_allclose(out, ref)


@pytest.mark.parametrize("func", ZERO_CHECKED, ids=lambda i: i.__name__)
def test_err_func_zeros(func: Callable[..., np.float64]) -> None:
    """Test the special-casing of all-zero input."""
    zeros = np.zeros_like(QM)
    assert func(zeros, zeros) == 0.0
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        assert func(QM, zeros) == np.inf
