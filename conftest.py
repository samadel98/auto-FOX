"""Shared pytest configuration."""

from __future__ import annotations

import pandas as pd
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Turn pandas 4 deprecations raised from within Auto-FOX into errors (pandas >= 3.1)."""
    if hasattr(pd.errors, "Pandas4Warning"):
        config.addinivalue_line("filterwarnings", r"error::pandas.errors.Pandas4Warning:FOX\.")
