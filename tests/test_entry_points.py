"""A module for testing :mod:`FOX.entry_points`."""

from __future__ import annotations

from typing import Any
from pathlib import Path

import pytest
from assertionlib import assertion

import FOX.entry_points
from FOX.entry_points import main_armc


@pytest.mark.parametrize("args,ref", [
    ([], False),
    (["-r"], True),
    (["-r", "True"], True),
    (["--restart", "true"], True),
    (["-r", "1"], True),
    (["-r", "False"], False),
    (["--restart", "no"], False),
], ids=["default", "flag", "True", "true", "1", "False", "no"])
def test_main_armc_restart(
    args: list[str], ref: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test the ``--restart`` argument of :func:`FOX.entry_points.main_armc`."""
    filename = tmp_path / "armc.yaml"
    filename.write_text("param: {}\n")

    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(FOX.entry_points, "dict_to_armc", lambda dct: (dct, {"a": 1}))
    monkeypatch.setattr(
        FOX.entry_points, "run_armc", lambda armc, **kwargs: calls.append((armc, kwargs))
    )

    main_armc([str(filename)] + args)
    assertion.eq(calls, [({"param": {}}, {"a": 1, "restart": ref})])


def test_main_armc_raise(tmp_path: Path) -> None:
    """Test that :func:`FOX.entry_points.main_armc` rejects invalid ``--restart`` values."""
    with pytest.raises(SystemExit):
        main_armc([str(tmp_path / "armc.yaml"), "-r", "bob"])
