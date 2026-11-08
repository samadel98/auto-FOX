"""Tests for :mod:`FOX.functions.msd` and the MSD methods of :class:`FOX.MultiMolecule`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from FOX import MultiMolecule, example_xyz
from FOX.functions.msd import get_msd, unwrap, get_diffusion_coefficient

MOL = MultiMolecule.from_xyz(example_xyz)

# A box with three non-orthogonal lattice vectors
LATTICE = np.array([
    [10.0, 0.0, 0.0],
    [3.0, 9.0, 0.0],
    [1.0, 2.0, 11.0],
])


def _random_walk(shape: tuple[int, int], sigma: float, seed: int = 0) -> np.ndarray:
    """Return a random walk of ``shape[1]`` atoms with steps of **sigma** per axis."""
    rng = np.random.default_rng(seed)
    return rng.normal(scale=sigma, size=(*shape, 3)).cumsum(axis=0)


def _msd_brute_force(xyz: np.ndarray) -> np.ndarray:
    """Calculate the MSD with an explicit loop over all lag times."""
    m = len(xyz)
    ret = np.empty(m)
    for tau in range(m):
        displacement = xyz[tau:] - xyz[:m - tau]
        ret[tau] = (displacement**2).sum(axis=-1).mean()
    return ret


class TestGetMSD:
    def test_brute_force(self) -> None:
        xyz = _random_walk((200, 7), sigma=1.0) + 50
        np.testing.assert_allclose(get_msd(xyz), _msd_brute_force(xyz), rtol=0, atol=1e-9)

    def test_chunks(self) -> None:
        xyz = _random_walk((50, 20), sigma=1.0)
        np.testing.assert_allclose(get_msd(xyz, nbytes_max=1000), get_msd(xyz), rtol=1e-12)

    def test_max_lag(self) -> None:
        xyz = _random_walk((50, 5), sigma=1.0)
        np.testing.assert_allclose(get_msd(xyz, max_lag=10), get_msd(xyz)[:11])
        assert get_msd(xyz, max_lag=0).tolist() == [0]
        assert len(get_msd(xyz, max_lag=100)) == 50
        with pytest.raises(ValueError):
            get_msd(xyz, max_lag=-1)

    def test_zero_lag(self) -> None:
        assert get_msd(MOL[:10])[0] == 0

    @pytest.mark.parametrize("xyz", [
        np.zeros((10, 3)),
        np.zeros((10, 2, 2)),
        np.zeros((0, 2, 3)),
        np.zeros((10, 0, 3)),
    ], ids=["2d", "2_axes", "no_frames", "no_atoms"])
    def test_raise(self, xyz: np.ndarray) -> None:
        with pytest.raises(ValueError):
            get_msd(xyz)


class TestUnwrap:
    def test_triclinic(self) -> None:
        xyz = _random_walk((300, 30), sigma=0.5)
        wrapped = ((xyz @ np.linalg.inv(LATTICE)) % 1) @ LATTICE
        ref = xyz - xyz[0] + wrapped[0]
        np.testing.assert_allclose(unwrap(wrapped, LATTICE), ref, atol=1e-10)

    def test_lattice_per_frame(self) -> None:
        xyz = _random_walk((100, 10), sigma=0.5)
        lattice = np.repeat(LATTICE[None], len(xyz), axis=0)
        np.testing.assert_allclose(unwrap(xyz, lattice), unwrap(xyz, LATTICE))

    def test_periodic_axes(self) -> None:
        # Only the x axis is periodic, so a jump of a whole box along z remains
        xyz = np.zeros((2, 1, 3))
        xyz[1, 0] = [10.0, 0.0, 11.0]
        lattice = np.diag([10.0, 9.0, 11.0])
        np.testing.assert_allclose(unwrap(xyz, lattice, periodic="x")[1, 0], [0.0, 0.0, 11.0])
        np.testing.assert_allclose(unwrap(xyz, lattice, periodic=[0, 2])[1, 0], [0.0, 0.0, 0.0])

    def test_raise(self) -> None:
        with pytest.raises(ValueError):
            unwrap(np.zeros((5, 2, 3)), np.zeros((4, 3, 3)))
        with pytest.raises(ValueError):
            unwrap(np.zeros((5, 2, 3)), LATTICE, periodic="w")


class TestInitMSD:
    def test_columns(self) -> None:
        df = MOL.init_msd(timestep=2.0, atom_subset=('Cd', 'Se', 'O'))
        assert df.columns.tolist() == ['Cd', 'Se', 'O']
        assert df.index.name == 'Time / fs'
        assert len(df) == len(MOL)
        np.testing.assert_allclose(df.index, 2.0 * np.arange(len(MOL)))

        xyz = np.asarray(MOL[:, MOL.atoms['Se']], dtype=np.float64)
        np.testing.assert_allclose(df['Se'], get_msd(xyz))

        df_short = MOL.init_msd(timestep=2.0, atom_subset=('Cd', 'Se', 'O'), max_lag=100)
        pd.testing.assert_frame_equal(df_short, df.iloc[:101])

    @pytest.mark.parametrize("atom_subset,columns", [
        (None, [0]),
        ('Cd', ['Cd']),
        (0, [0]),
        ([0, 1, 2], [0]),
        ([[0, 1], [2, 3]], [0, 1]),
    ])
    def test_atom_subset(self, atom_subset: object, columns: list[object]) -> None:
        df = MOL.init_msd(atom_subset=atom_subset, mol_subset=slice(0, 50))
        assert df.columns.tolist() == columns
        assert df.shape == (50, len(columns))

    def test_periodic(self) -> None:
        xyz = _random_walk((300, 40), sigma=0.5)
        wrapped = ((xyz @ np.linalg.inv(LATTICE)) % 1) @ LATTICE
        mol = MultiMolecule(wrapped, atoms={'Ar': list(range(40))})
        mol.lattice = LATTICE
        df = mol.init_msd(periodic='xyz')
        np.testing.assert_allclose(df[0], get_msd(xyz), rtol=1e-9, atol=1e-9)

        # Without unwrapping, atoms crossing the box seem to jump a whole box length
        assert mol.init_msd().iloc[1, 0] > 2 * df.iloc[1, 0]

    def test_periodic_no_lattice(self) -> None:
        with pytest.raises(TypeError):
            MOL.init_msd(periodic='xyz')


class TestDiffusionCoefficient:
    def test_random_walk(self) -> None:
        # The diffusion coefficient of a random walk is sigma**2 / (2 * timestep) per axis
        sigma, timestep = 0.05, 2.0
        mol = MultiMolecule(_random_walk((2000, 500), sigma), atoms={'Ar': list(range(500))})
        ret = mol.init_diffusion_coefficient(timestep=timestep, atom_subset='Ar')
        assert ret.index.tolist() == ['Ar']
        ref = sigma**2 / (2 * timestep) * 1e-5  # Å^2/fs to m^2/s
        np.testing.assert_allclose(ret['Ar'], ref, rtol=0.05)

    def test_fit_range(self) -> None:
        # A line with a slope of 6 Å^2/fs, i.e. D = 1 Å^2/fs, except before 10 fs
        t = np.arange(100, dtype=np.float64)
        msd = pd.Series(np.where(t < 10, 0.0, 6 * t), index=t)
        np.testing.assert_allclose(get_diffusion_coefficient(msd, start=10).iloc[0], 1e-5)
        np.testing.assert_allclose(get_diffusion_coefficient(msd, ndim=2, start=10).iloc[0], 1.5e-5)
        assert not np.isclose(get_diffusion_coefficient(msd, start=0).iloc[0], 1e-5)

    def test_raise(self) -> None:
        msd = pd.Series(np.arange(10.0), index=np.arange(10.0))
        with pytest.raises(ValueError):
            get_diffusion_coefficient(msd, start=5, stop=5.5)
        with pytest.raises(ValueError):
            get_diffusion_coefficient(msd.iloc[:0])
