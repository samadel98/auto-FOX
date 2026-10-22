"""Compatibility utilities for supporting multiple NumPy versions."""

from __future__ import annotations

import numpy as np

__all__ = ["COPY_IF_NEEDED"]

#: The value of the ``copy`` parameter in :func:`np.array` for only copying when needed.
#: This is ``False`` in NumPy 1.x, but NumPy 2 changed its meaning to "never copy"
#: in favor of ``None``.
COPY_IF_NEEDED: bool | None = None if np.lib.NumpyVersion(np.__version__) >= "2.0.0" else False
