"""A module for testing :mod:`FOX.entry_points`."""

from __future__ import annotations

import os
import shutil
from typing import Any
from pathlib import Path

import h5py
import pytest
import yaml
import numpy as np
import pandas as pd
from assertionlib import assertion
from FOXdata import ARMC_DIR as ARMC_REF, ARMCPT_DIR as ARMCPT_REF

import FOX.entry_points
from FOX._nanoutils import UniqueLoader
from FOX.armc import dict_to_armc
from FOX.recipes import get_best
from FOX.recipes.param import PLT_ERROR
from FOX.entry_points import main_armc, main_armc2yaml, main_armc_report

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


@pytest.mark.parametrize("ref_dir", [ARMC_REF, ARMCPT_REF], ids=["armc", "armcpt"])
def test_main_armc_report(
    ref_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test :func:`FOX.entry_points.main_armc_report`."""
    filename = tmp_path / "armc.hdf5"
    shutil.copyfile(ref_dir / "armc.hdf5", filename)

    main_armc_report([str(filename)])
    out = capsys.readouterr().out
    assertion.contains(out, "Finished iterations: 100 of 100")
    assertion.contains(out, "Best parameters:")

    # Without `--output` the files are written next to the .hdf5 file
    param = pd.read_csv(tmp_path / "report" / "best_param.csv", index_col=[0, 1, 2]).squeeze()
    param_ref = get_best(filename, "param")
    np.testing.assert_allclose(param.values, param_ref.values)
    if ref_dir == ARMCPT_REF:
        # The best of the 3 parameter sets
        assertion.contains(out, "(super-iteration 9, sub-iteration 2, parameter set 0)")
    if PLT_ERROR is None:
        # Only the PES descriptors of the best parameter set are plotted
        names = ["aux_error.png", "best_param.csv", "rdf.0.png"]
        assertion.eq(sorted(os.listdir(tmp_path / "report")), names)

    # Only the .csv file with `--no-plots`
    output = tmp_path / "out"
    main_armc_report([str(filename), "-o", str(output), "--no-plots", "--validation"])
    assertion.eq(sorted(os.listdir(output)), ["best_param.csv"])
    assertion.contains(capsys.readouterr().out, "Lowest validation error")


def test_main_armc_report_values(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test the values printed by :func:`FOX.entry_points.main_armc_report`."""
    filename = tmp_path / "armc.hdf5"
    shutil.copyfile(ARMC_REF / "armc.hdf5", filename)
    main_armc_report([str(filename), "--no-plots"])
    out = capsys.readouterr().out
    assertion.contains(out, "Acceptance rate: 74.0%")
    assertion.contains(out, "Lowest error: 0.0732069 (super-iteration 5, sub-iteration 1)")


def test_main_armc_report_raise(tmp_path: Path) -> None:
    """Test that :func:`FOX.entry_points.main_armc_report` raises for unusable files."""
    with pytest.raises(FileNotFoundError):
        main_armc_report([str(tmp_path / "armc.hdf5")])

    filename = tmp_path / "armc.hdf5"
    shutil.copyfile(ARMC_REF / "armc.hdf5", filename)
    with h5py.File(filename, "r+") as f:
        f.attrs["super-iteration"] = -1
        f.attrs["sub-iteration"] = -1
    with pytest.raises(ValueError):
        main_armc_report([str(filename)])
