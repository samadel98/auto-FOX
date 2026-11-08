"""Tests for the top-level :mod:`FOX` package."""

from __future__ import annotations

import sys
import subprocess

SCRIPT = """
import warnings
import pandas as pd
import FOX

with warnings.catch_warnings(record=True) as record:
    for module in ["FOX", "FOX.armc.param_mapping", "FOXdata", "my_script"]:
        warnings.warn_explicit(module, pd.errors.PerformanceWarning, "file.py", 1, module=module)
print(" ".join(str(w.message) for w in record))
"""


def test_performance_warning_filter() -> None:
    """Test that ``import FOX`` only hides pandas performance warnings raised from Auto-FOX."""
    # Run in a new interpreter: pytest restores the warning filters after collecting the tests,
    # which drops the filters added by ``import FOX``
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT], check=True, capture_output=True, text=True
    ).stdout
    assert out.split() == ["FOXdata", "my_script"]
