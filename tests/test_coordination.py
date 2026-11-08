"""Tests for the coordination-number methods of :class:`FOX.MultiMolecule`."""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

import FOX.classes.multi_mol
from FOX import MultiMolecule, example_xyz

MOL = MultiMolecule.from_xyz(example_xyz)[:50]
MOL.setflags(write=False)


def _brute_force(mol: MultiMolecule, at1: str, at2: str, r_max: float) -> np.ndarray:
    """Count the neighbours with an explicit loop over all molecules."""
    i, j = mol.atoms[at1], mol.atoms[at2]
    ret = []
    for xyz in mol:
        dist = cdist(xyz[i], xyz[j])
        dist[np.equal.outer(i, j)] = np.inf
        ret.append((dist <= r_max).sum(axis=1))
    return np.array(ret)


def _line_mol() -> MultiMolecule:
    """One Cd atom with O atoms at 1, 2 and 3 Å along x, in a 10 Å box."""
    xyz = np.array([[[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [9.5, 0, 0]]])
    mol = MultiMolecule(xyz, atoms={'Cd': [0], 'O': [1, 2, 3, 4]})
    mol.lattice = np.eye(3) * 10
    return mol


class TestGetCoordinationNumber:
    @pytest.mark.parametrize("pair,r_max", [
        (("Cd", "O"), 3.0),
        (("Cd", "Se"), 3.0),
        (("O", "Cd"), 2.5),
        (("Cd", "Cd"), 4.5),
    ])
    def test_brute_force(self, pair: tuple[str, str], r_max: float) -> None:
        ret = MOL.get_coordination_number(pair, r_max)
        np.testing.assert_array_equal(ret, _brute_force(MOL, *pair, r_max))
        assert ret.shape == (len(MOL), len(MOL.atoms[pair[0]]))

    def test_distances(self) -> None:
        mol = _line_mol()
        assert mol.get_coordination_number(("Cd", "O"), 2.5).tolist() == [[2]]
        assert mol.get_coordination_number(("Cd", "O"), 3.0).tolist() == [[3]]
        # The O atom at 9.5 Å is 0.5 Å away through the periodic boundary
        assert mol.get_coordination_number(("Cd", "O"), 1.0, periodic="x").tolist() == [[2]]
        assert mol.get_coordination_number(("Cd", "O"), 1.0, periodic="yz").tolist() == [[1]]

    def test_alias(self) -> None:
        mol = MOL.copy()
        mol.atoms_alias = {"Cd2": ("Cd", np.s_[:10])}
        ret = mol.get_coordination_number(("Cd2", "O"), 3.0)
        np.testing.assert_array_equal(ret, MOL.get_coordination_number(("Cd", "O"), 3.0)[:, :10])

    def test_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ref = MOL.get_coordination_number(("Cd", "Se"), 3.0)
        slice_iter = functools.partial(FOX.classes.multi_mol.slice_iter, nbytes_max=100_000)
        monkeypatch.setattr(FOX.classes.multi_mol, "slice_iter", slice_iter)
        np.testing.assert_array_equal(MOL.get_coordination_number(("Cd", "Se"), 3.0), ref)

    @pytest.mark.parametrize("pair,kwargs,exc", [
        (("Cd", "Bob"), {}, ValueError),
        (("Cd", "O"), {"periodic": "xyz"}, TypeError),
        (("Cd", "O"), {"r_max": 0}, ValueError),
    ])
    def test_raise(self, pair: tuple[str, str], kwargs: dict[str, Any], exc: type) -> None:
        kwargs.setdefault("r_max", 3.0)
        with pytest.raises(exc):
            MOL.get_coordination_number(pair, **kwargs)


class TestInitCoordinationNumber:
    def test_passes(self) -> None:
        df = MOL.init_coordination_number([("Cd", "O"), ("Cd", "Se")], r_max=3.0)
        assert df.columns.tolist() == ["Cd O", "Cd Se"]
        assert df.index.name == "XYZ frame number"
        assert len(df) == len(MOL)
        ref = _brute_force(MOL, "Cd", "Se", 3.0).mean(axis=1)
        np.testing.assert_allclose(df["Cd Se"], ref)

    def test_r_max_mapping(self) -> None:
        df = MOL.init_coordination_number(
            [("Cd", "O"), ("Cd", "Se")], r_max={"Cd O": 2.5, "Cd Se": 3.0}
        )
        np.testing.assert_allclose(df["Cd O"], _brute_force(MOL, "Cd", "O", 2.5).mean(axis=1))
        np.testing.assert_allclose(df["Cd Se"], _brute_force(MOL, "Cd", "Se", 3.0).mean(axis=1))

    def test_iterator(self) -> None:
        df1 = MOL.init_coordination_number(iter([("Cd", "O"), ("Cd", "Se")]), r_max=3.0)
        df2 = MOL.init_coordination_number([["Cd", "O"], ["Cd", "Se"]], r_max=3.0)
        pd.testing.assert_frame_equal(df1, df2)

    @pytest.mark.parametrize("atom_pairs,r_max,exc", [
        (("Cd", "O"), 3.0, TypeError),
        ("Cd", 3.0, TypeError),
        ([("Cd", "O")], {"Cd Se": 3.0}, KeyError),
    ], ids=["single_pair", "str", "missing_r_max"])
    def test_raise(self, atom_pairs: Any, r_max: Any, exc: type) -> None:
        with pytest.raises(exc):
            MOL.init_coordination_number(atom_pairs, r_max=r_max)


class TestInitCoordinationDistribution:
    def test_passes(self) -> None:
        df = MOL.init_coordination_distribution([("Cd", "O"), ("Cd", "Se")], r_max=3.0)
        assert df.columns.tolist() == ["Cd O", "Cd Se"]
        assert df.index.name == "Coordination number"
        np.testing.assert_allclose(df.sum(), 1)

        cn = _brute_force(MOL, "Cd", "Se", 3.0)
        assert df.index.tolist() == list(range(1 + max(cn.max(), df.index.max())))
        ref = np.bincount(cn.ravel(), minlength=len(df)) / cn.size
        np.testing.assert_allclose(df["Cd Se"], ref)

    def test_cn_max(self) -> None:
        df = MOL.init_coordination_distribution([("Cd", "Se")], r_max=3.0)
        df_short = MOL.init_coordination_distribution([("Cd", "Se")], r_max=3.0, cn_max=2)
        assert df_short.index.tolist() == [0, 1, 2]
        np.testing.assert_allclose(df_short["Cd Se"].iloc[:2], df["Cd Se"].iloc[:2])
        np.testing.assert_allclose(df_short["Cd Se"].iloc[2], df["Cd Se"].iloc[2:].sum())

        df_long = MOL.init_coordination_distribution([("Cd", "Se")], r_max=3.0, cn_max=10)
        assert len(df_long) == 11
        np.testing.assert_allclose(df_long["Cd Se"].iloc[:len(df)], df["Cd Se"])

        with pytest.raises(ValueError):
            MOL.init_coordination_distribution([("Cd", "Se")], r_max=3.0, cn_max=-1)
