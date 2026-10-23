"""A module for testing :mod:`FOX.entry_points`."""

from __future__ import annotations

import os
from typing import Any
from pathlib import Path

import pytest
import yaml
from assertionlib import assertion
from nanoutils import UniqueLoader

import FOX.entry_points
from FOX.armc import dict_to_armc
from FOX.entry_points import main_armc, main_armc2yaml

PATH = Path("tests") / "test_files"


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


def test_main_armc2yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test :func:`FOX.entry_points.main_armc2yaml`."""
    # The input has a `!!python/object/apply:builtins.slice` tag, which `init_armc` accepts
    with open(PATH / "armc_ref.yaml", "r") as f:
        dct = yaml.load(f, Loader=UniqueLoader)
    dct["monte_carlo"].update(path=str(tmp_path), seed=5)
    dct["psf"]["str_file"] = os.path.abspath(dct["psf"]["str_file"])
    dct["job"]["molecule"] = os.path.abspath(dct["job"]["molecule"])
    dct["job"]["md"]["settings"]["prm"] = os.path.abspath(dct["job"]["md"]["settings"]["prm"])
    filename = tmp_path / "armc.yaml"
    filename.write_text(yaml.dump(dct))

    # Without `--output` the file is written to the working directory, without overwriting
    monkeypatch.chdir(tmp_path)
    main_armc2yaml([str(filename)])
    main_armc2yaml([str(filename)])
    assert (tmp_path / "armc.0.yaml").is_file()
    assert (tmp_path / "armc.1.yaml").is_file()

    with open(tmp_path / "armc.1.yaml", "r") as f:
        dct_out = yaml.load(f, Loader=UniqueLoader)
    assertion.eq(dct_out["monte_carlo"]["seed"], 5)
    armc, _ = dict_to_armc(dct_out)
    assertion.eq(armc.seed, 5)


def test_main_armc2yaml_raise(tmp_path: Path) -> None:
    """Test that :func:`FOX.entry_points.main_armc2yaml` raises for a missing input file."""
    with pytest.raises(FileNotFoundError):
        main_armc2yaml([str(tmp_path / "armc.yaml"), "-o", str(tmp_path / "out.yaml")])
