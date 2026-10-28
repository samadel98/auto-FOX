"""A module with the :class:`StopCriteria` class.

Index
-----
.. currentmodule:: FOX.armc
.. autosummary::
    StopCriteria

API
---
.. autoclass:: StopCriteria
    :members:

"""

from __future__ import annotations

import math
import operator
from typing import Any, Dict, Optional, SupportsFloat, SupportsIndex

__all__ = ['StopCriteria']


def _as_float(name: str, value: SupportsFloat, positive: bool) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name!r} expected a float; observed type: 'bool'")
    ret = float(value)
    if not math.isfinite(ret) or ret < 0 or (positive and ret == 0):
        sign = "positive" if positive else "non-negative"
        raise ValueError(f"{name!r} expected a finite {sign} float; observed value: {value!r}")
    return ret


class StopCriteria:
    """The criteria for stopping a Monte Carlo optimization before it runs out of iterations.

    All criteria are optional; the optimization stops as soon as any of them is met.

    Parameters
    ----------
    target_error : :class:`float`, optional
        Stop once the lowest error found so far is at or below this value.
        The error of a parameter set is its total (weighted) auxiliary error,
        *i.e.* the sum over all PES descriptors and molecules.
    patience : :class:`int`, optional
        Stop after this many super-iterations in a row without a new lowest error.
    max_hours : :class:`float`, optional
        Stop once the optimization has run for this many hours.
        The clock starts anew when a calculation is restarted.

    """

    __slots__ = ('target_error', 'patience', 'max_hours')

    target_error: Optional[float]
    patience: Optional[int]
    max_hours: Optional[float]

    def __init__(
        self,
        target_error: Optional[SupportsFloat] = None,
        patience: Optional[SupportsIndex] = None,
        max_hours: Optional[SupportsFloat] = None,
    ) -> None:
        if target_error is None:
            self.target_error = None
        else:
            self.target_error = _as_float('target_error', target_error, positive=False)

        if patience is None:
            self.patience = None
        elif isinstance(patience, bool):
            raise TypeError("'patience' expected an integer; observed type: 'bool'")
        else:
            self.patience = operator.index(patience)
            if self.patience < 1:
                raise ValueError(f"'patience' expected a positive integer; "
                                 f"observed value: {patience!r}")

        if max_hours is None:
            self.max_hours = None
        else:
            self.max_hours = _as_float('max_hours', max_hours, positive=True)

    def __repr__(self) -> str:
        """Implement :code:`repr(self)`."""
        args = ', '.join(f'{k}={v!r}' for k, v in self.to_yaml_dict().items())
        return f'{type(self).__name__}({args})'

    def __eq__(self, value: Any) -> bool:
        """Implement :code:`self == value`."""
        if type(self) is not type(value):
            return NotImplemented
        return self.to_yaml_dict() == value.to_yaml_dict()

    def __bool__(self) -> bool:
        """Return whether any of the criteria is set."""
        return any(v is not None for v in self.to_yaml_dict().values())

    def to_yaml_dict(self) -> Dict[str, Any]:
        """Convert this instance into a .yaml-compatible dictionary."""
        return {
            'target_error': self.target_error,
            'patience': self.patience,
            'max_hours': self.max_hours,
        }

    def check_iteration(self, best_error: float, hours: float) -> Optional[str]:
        """Check the criteria that apply after every iteration.

        Returns the reason for stopping, or :data:`None` if the optimization should continue.

        """
        if self.target_error is not None and best_error <= self.target_error:
            return (f"the lowest error ({best_error:.6g}) reached "
                    f"stop.target_error ({self.target_error:.6g})")
        if self.max_hours is not None and hours >= self.max_hours:
            return f"the run reached stop.max_hours ({self.max_hours:g} hours)"
        return None

    def check_super_iteration(self, n_without_improvement: int) -> Optional[str]:
        """Check the criteria that apply at the end of every super-iteration.

        Returns the reason for stopping, or :data:`None` if the optimization should continue.

        """
        if self.patience is not None and n_without_improvement >= self.patience:
            return (f"the lowest error did not improve for {n_without_improvement} "
                    f"super-iterations (stop.patience = {self.patience})")
        return None
