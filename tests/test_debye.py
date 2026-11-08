"""Tests for :mod:`FOX.functions.debye` and :meth:`FOX.MultiMolecule.init_debye_scattering`."""

from __future__ import annotations

import functools

import numpy as np
import pytest
from scipy.spatial.distance import cdist

import FOX.classes.multi_mol
from FOX import MultiMolecule, example_xyz
from FOX.functions.debye import get_debye_scattering, _get_scattering

MOL = MultiMolecule.from_xyz(example_xyz)[:3]

#: The Cu K-alpha wavelength in nm
WAVELENGTH = 0.15406


def _get_q(half_angle: float | np.ndarray, wavelength: float = WAVELENGTH) -> np.ndarray:
    """Return the scattering vector length in Angstrom**-1; **wavelength** is in nm."""
    return np.atleast_1d(4 * np.pi * np.sin(half_angle) / (10 * wavelength))


def _debye_brute_force(xyz1: np.ndarray, xyz2: np.ndarray, f1: float, f2: float, q: float) -> float:
    """Calculate the Debye scattering of a single molecule with an explicit loop."""
    ret = 0.0
    for r in cdist(xyz1, xyz2).ravel():
        if r != 0:
            ret += f1 * f2 * np.sin(q * r) / (q * r)
    return ret


def _diatomic(d: float, symbols: tuple[str, str] = ("Cd", "Se")) -> MultiMolecule:
    """Return a molecule with two atoms at a distance of **d** Angstrom."""
    xyz = np.array([[[0.0, 0.0, 0.0], [d, 0.0, 0.0]]])
    if symbols[0] == symbols[1]:
        return MultiMolecule(xyz, atoms={symbols[0]: [0, 1]})
    return MultiMolecule(xyz, atoms={symbols[0]: [0], symbols[1]: [1]})


class TestScatteringFactors:
    def test_forward_scattering(self) -> None:
        # At q = 0 the scattering factor equals the number of electrons
        f = _get_scattering(np.array(["H", "O", "Se", "Cd"]), np.array([0.0]))
        np.testing.assert_allclose(f[:, 0], [1, 8, 34, 48], atol=0.02)

    def test_shape(self) -> None:
        q = np.linspace(0, 5, 7)
        f = _get_scattering(np.array(["Cd", "Se", "O"]), q)
        assert f.shape == (3, 7)
        np.testing.assert_array_equal(f[1], _get_scattering(np.array(["Se"]), q)[0])
        assert (np.diff(f, axis=1) < 0).all()


class TestGetDebyeScattering:
    def test_shape(self) -> None:
        dist_mat = np.random.default_rng(0).uniform(1, 10, size=(4, 3, 2))
        q = np.array([0.5, 1.0, 2.0])
        ret = get_debye_scattering(dist_mat, np.array(["Cd"] * 3), np.array(["Se"] * 2), q)
        assert ret.shape == (4, 3)
        for k, q_k in enumerate(q):
            ref = get_debye_scattering(dist_mat, ["Cd"] * 3, ["Se"] * 2, [q_k])
            np.testing.assert_allclose(ret[:, k], ref[:, 0])

    def test_zero_q(self) -> None:
        # sin(qr) / qr = 1 for q = 0; zero distances (an atom and itself) are skipped
        dist_mat = np.array([[[0.0, 2.0], [2.0, 0.0]]])
        ret = get_debye_scattering(dist_mat, ["Cd", "Cd"], ["Cd", "Cd"], [0.0])
        f = _get_scattering(np.array(["Cd"]), np.array([0.0]))[0, 0]
        np.testing.assert_allclose(ret, [[2 * f**2]])

    def test_nan(self) -> None:
        dist_mat = np.array([[[np.nan, 2.0]]])
        ret = get_debye_scattering(dist_mat, ["Cd"], ["Se", "Se"], [1.0])
        ref = get_debye_scattering(dist_mat[..., 1:], ["Cd"], ["Se"], [1.0])
        np.testing.assert_allclose(ret, ref)

    @pytest.mark.parametrize("kwargs", [
        {"dist_mat": np.ones((1, 1, 1, 1))},
        {"symbols1": np.array([["Cd"]])},
        {"scattering_vector": np.ones((1, 1))},
    ])
    def test_raise(self, kwargs: dict[str, np.ndarray]) -> None:
        params = {
            "dist_mat": np.ones((1, 1, 1)),
            "symbols1": np.array(["Cd"]),
            "symbols2": np.array(["Se"]),
            "scattering_vector": np.ones(1),
        }
        params.update(kwargs)
        with pytest.raises(ValueError):
            get_debye_scattering(**params)


class TestInitDebyeScattering:
    @pytest.mark.parametrize("d", [1.0, 2.6, 7.5])
    def test_diatomic(self, d: float) -> None:
        # The textbook Debye equation for a single pair of atoms
        half_angle = 0.3
        q = _get_q(half_angle)
        df = _diatomic(d).init_debye_scattering(half_angle, WAVELENGTH, atom_subset=["Cd", "Se"])
        f_cd, f_se = _get_scattering(np.array(["Cd", "Se"]), q)[:, 0]

        np.testing.assert_allclose(df.index, q)
        assert df.index.name == "Q  /  Angstrom**-1"
        assert df.columns.tolist() == ["Cd Cd", "Cd Se", "Se Se"]
        np.testing.assert_allclose(df["Cd Se"], f_cd * f_se * np.sin(q * d) / (q * d))
        np.testing.assert_array_equal(df[["Cd Cd", "Se Se"]], 0)

    def test_identical_atoms(self) -> None:
        # Both orders (i, j) and (j, i) of a pair of identical elements are counted
        q = _get_q(0.4)
        df = _diatomic(3.0, ("Cd", "Cd")).init_debye_scattering(0.4, WAVELENGTH)
        f = _get_scattering(np.array(["Cd"]), q)[0, 0]
        np.testing.assert_allclose(df["Cd Cd"], 2 * f**2 * np.sin(q * 3.0) / (q * 3.0))

    def test_cu_kalpha(self) -> None:
        # 2 theta = 30 degrees with Cu K-alpha radiation
        df = _diatomic(2.6).init_debye_scattering(np.radians(15), WAVELENGTH)
        np.testing.assert_allclose(df.index, [2.11114], rtol=1e-5)

    def test_angles(self) -> None:
        half_angle = np.linspace(0, np.pi / 2, 5)
        df = MOL.init_debye_scattering(half_angle, WAVELENGTH, atom_subset=["Cd", "Se"])
        assert df.shape == (5, 3)
        np.testing.assert_allclose(df.index, _get_q(half_angle))
        for k, theta in enumerate(half_angle):
            ref = MOL.init_debye_scattering(theta, WAVELENGTH, atom_subset=["Cd", "Se"])
            np.testing.assert_allclose(df.iloc[k], ref.iloc[0])

    def test_wavelengths(self) -> None:
        wavelength = np.array([0.07107, 0.15406])  # Mo and Cu K-alpha
        df = MOL.init_debye_scattering(0.3, wavelength, atom_subset=["Cd", "Se"])
        np.testing.assert_allclose(df.index, _get_q(0.3, wavelength))

    @pytest.mark.parametrize("half_angle", [0.1, 0.5])
    def test_brute_force(self, half_angle: float) -> None:
        q = _get_q(half_angle)[0]
        df = MOL.init_debye_scattering(half_angle, WAVELENGTH, atom_subset=["Cd", "Se"])
        f_cd, f_se = _get_scattering(np.array(["Cd", "Se"]), np.array([q]))[:, 0]
        cd, se = MOL.atoms["Cd"], MOL.atoms["Se"]

        ref = np.mean([_debye_brute_force(xyz[cd], xyz[se], f_cd, f_se, q) for xyz in MOL])
        np.testing.assert_allclose(df["Cd Se"].iloc[0], ref)
        ref = np.mean([_debye_brute_force(xyz[cd], xyz[cd], f_cd, f_cd, q) for xyz in MOL])
        np.testing.assert_allclose(df["Cd Cd"].iloc[0], ref)

    def test_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        half_angle = np.array([0.1, 0.3, 0.5])
        ref = MOL.init_debye_scattering(half_angle, WAVELENGTH, atom_subset=["Cd", "Se"])
        slice_iter = functools.partial(FOX.classes.multi_mol.slice_iter, nbytes_max=50_000)
        monkeypatch.setattr(FOX.classes.multi_mol, "slice_iter", slice_iter)
        df = MOL.init_debye_scattering(half_angle, WAVELENGTH, atom_subset=["Cd", "Se"])
        np.testing.assert_allclose(df, ref)

    def test_raise(self) -> None:
        with pytest.raises(ValueError):
            MOL.init_debye_scattering(np.ones((2, 2)), WAVELENGTH)
