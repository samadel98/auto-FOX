"""Tests for various ARMC jobs."""

import os
import warnings
import pytest
import shutil
import subprocess
from os import PathLike
from pathlib import Path
from typing import Tuple, Generator, Any, cast, List, Optional, Union
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import h5py
import yaml
from schema import SchemaError
from nanoutils import delete_finally, UniqueLoader
from assertionlib import assertion

from FOX.testing_utils import load_results, compare_hdf5
from FOX.armc import dict_to_armc, run_armc
from FOX.armc.sanitization import _sort_atoms
from FOXdata import (
    ARMC_DIR as ARMC_REF,
    ARMCPT_DIR as ARMCPT_REF,
)

PATH = Path('tests') / 'test_files'

DF = pd.DataFrame({
    "param_type": ['charge', 'charge', 'charge', 'charge', 'epsilon', 'epsilon', 'epsilon', 'epsilon', 'epsilon', 'sigma', 'sigma', 'sigma', 'sigma', 'sigma'],  # noqa: E501
    "atoms": ['Cd', 'Se', 'O_1', 'C_1', 'Cd Cd', 'Se Se', 'Cd Se', 'Cd O_1', 'O_1 Se', 'Cd Cd', 'Se Se', 'Cd Se', 'Cd O_1', 'O_1 Se'],  # noqa: E501
})


def _prepare_restart(
    inp: Union[str, "PathLike[str]"],
    out: Union[str, "PathLike[str]"],
    super_iter: Optional[int] = None,
    sub_iter: Optional[int] = None
) -> None:
    shutil.copytree(inp, out)
    hdf5 = Path(out) / 'armc.hdf5'
    with h5py.File(hdf5, 'r+') as f:
        if super_iter is not None:
            f.attrs['super-iteration'] = super_iter
        if sub_iter is not None:
            f.attrs['sub-iteration'] = sub_iter


@delete_finally(PATH / '_ARMC')
@pytest.mark.parametrize('name', ['armc_ref.yaml'])
def test_armc_restart(name: str) -> None:
    """Test the restart functionality of :class:`ARMC` and :class:`ARMCPT`."""
    file = PATH / name
    with open(file, 'r') as f:
        dct = yaml.load(f.read(), Loader=UniqueLoader)

    sub_iter = 5
    super_iter = 8
    i = 2 + (super_iter * 10 + sub_iter)

    armc, job_kwargs = dict_to_armc(dct)
    if name == 'armc_ref.yaml':
        REF = ARMC_REF
        armc.package_manager.hook = iter(load_results(ARMC_REF, n=1)[i:])
    else:
        iterator = _get_phi_iter()

        def swap_phi(*args: Any, **kwargs: Any) -> List[Tuple[int, int]]:
            return next(iterator)

        REF = ARMCPT_REF
        armc.swap_phi = swap_phi
        armc.package_manager.hook = iter(load_results(ARMCPT_REF, n=3)[i:])

    _prepare_restart(REF, PATH / job_kwargs['folder'], super_iter, sub_iter)
    hdf5_ref = REF / 'armc.hdf5'
    hdf5 = PATH / job_kwargs['folder'] / 'armc.hdf5'

    run_armc(armc, restart=True, **job_kwargs)
    assertion.assert_(next, armc.package_manager.hook, exception=StopIteration)
    with h5py.File(hdf5, 'r') as f1, h5py.File(hdf5_ref, 'r') as f2:
        compare_hdf5(f1, f2, skip={'/param', '/aux_error_mod'})
        assertion.shape_eq(f1['param'], f2['param'])
        assertion.shape_eq(f1['aux_error_mod'], f2['aux_error_mod'])
        assertion.eq(f1['param'].dtype, f2['param'].dtype)
        assertion.eq(f1['aux_error_mod'].dtype, f2['aux_error_mod'].dtype)


def test_armc_guess() -> None:
    file = PATH / 'armc_ref.yaml'
    with open(file, 'r') as f:
        dct = yaml.load(f.read(), Loader=UniqueLoader)

    sigma = dct['param']['lennard_jones'][1]
    del sigma['Cd Cd']
    sigma['frozen'] = {'guess': 'uff'}

    armc, job_kwargs = dict_to_armc(dct)
    param = armc.param.param.loc[('lennard_jones', 'sigma'), 0]

    # The expected `sigma` parameters
    ref = np.array([
        0.30504372,
        0.3654912,
        0.25372795,
        0.20704486,
        0.2471,
        0.294,
        0.26749234,
        0.3526,
        0.4852,
    ])
    np.testing.assert_allclose(param.values, ref)


def test_allow_non_existent() -> None:
    """Test ``param.validation.allow_non_existent``."""
    file = PATH / 'armc_ref.yaml'
    with open(file, 'r') as f:
        dct = yaml.load(f.read(), Loader=UniqueLoader)

    # Test `param.validation.allow_non_existent`
    dct['param']['charge']['Cl'] = -1
    assertion.assert_(dict_to_armc, dct, exception=RuntimeError)

    del dct['psf']
    del dct['param']['charge']['constraints'][0]
    assertion.assert_(dict_to_armc, dct, exception=RuntimeError)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dct['param']['validation']['allow_non_existent'] = True
        dct['param']['validation']['charge_tolerance'] = 'inf'
        assertion.assert_(dict_to_armc, dct)


def test_charge_tolerance() -> None:
    """Test ``param.validation.charge_tolerance``."""
    file = PATH / 'armc_ref.yaml'
    with open(file, 'r') as f:
        dct = yaml.load(f.read(), Loader=UniqueLoader)

    dct = dct.copy()
    dct['param']['charge']['Cd'] = 2
    assertion.assert_(dict_to_armc, dct, exception=ValueError)

    dct['param']['validation']['charge_tolerance'] = 'inf'
    assertion.assert_(dict_to_armc, dct)


def test_enforce_constraints() -> None:
    """Test ``param.validation.enforce_constraints``."""
    file = PATH / 'armc_ref.yaml'
    with open(file, 'r') as f:
        dct = yaml.load(f.read(), Loader=UniqueLoader)
        dct['param']['validation']['enforce_constraints'] = True
        dct['param']['validation']['charge_tolerance'] = 'inf'

    dct['param']['charge']['CG2O3'] = 10
    assertion.assert_(dict_to_armc, dct, exception=RuntimeError)

    dct['param']['charge']['CG2O3'] = -10
    assertion.assert_(dict_to_armc, dct, exception=RuntimeError)

    dct['param']['charge']['CG2O3'] = 0.4524
    dct['param']['charge']['Cd'] = 2
    assertion.assert_(dict_to_armc, dct, exception=RuntimeError)

    with pytest.warns(RuntimeWarning):
        dct['param']['validation']['enforce_constraints'] = False
        _ = dict_to_armc(dct)


def _get_phi_iter(n: int = 3) -> Generator[List[Tuple[int, int]], None, None]:
    while True:
        iterator = combinations_with_replacement(range(n), r=2)
        for i in iterator:
            yield cast(List[Tuple[int, int]], [i])


@delete_finally(PATH / '_ARMC', PATH / '_ARMCPT')
@pytest.mark.parametrize('name', ['armc_ref.yaml', 'armcpt_ref.yaml'])
def test_armc(name: str) -> None:
    """Test :class:`ARMC` and :class:`ARMCPT`."""
    file = PATH / name
    with open(file, 'r') as f:
        dct = yaml.load(f.read(), Loader=UniqueLoader)
    dct['monte_carlo']['seed'] = 0

    armc, job_kwargs = dict_to_armc(dct)
    if name == 'armc_ref.yaml':
        REF = ARMC_REF
        armc.package_manager.hook = iter(load_results(ARMC_REF, n=1))
    else:
        iterator = _get_phi_iter()

        def swap_phi(*args: Any, **kwargs: Any) -> List[Tuple[int, int]]:
            return next(iterator)

        REF = ARMCPT_REF
        armc.swap_phi = swap_phi
        armc.package_manager.hook = iter(load_results(ARMCPT_REF, n=3))

    hdf5_ref = REF / 'armc.hdf5'
    hdf5 = PATH / job_kwargs['folder'] / 'armc.hdf5'

    run_armc(armc, restart=False, **job_kwargs)
    assertion.assert_(next, armc.package_manager.hook, exception=StopIteration)
    with h5py.File(hdf5, 'r') as f1, h5py.File(hdf5_ref, 'r') as f2:
        compare_hdf5(f1, f2, skip={'/param', '/aux_error_mod'})
        assertion.shape_eq(f1['param'], f2['param'])
        assertion.shape_eq(f1['aux_error_mod'], f2['aux_error_mod'])
        assertion.eq(f1['param'].dtype, f2['param'].dtype)
        assertion.eq(f1['aux_error_mod'].dtype, f2['aux_error_mod'].dtype)

    # Validate that the non-charges are updated independently of each other
    with h5py.File(hdf5, 'r') as f1:
        _index = f1['param'].attrs['index'][0] != b"charge"
        index = np.nonzero(_index)[0]
        param = f1['param'][..., index]

        grad = param[:, 1:] - param[:, :-1]
        grad[grad < 1e-8] = 0
        assertion.le(np.count_nonzero(grad, axis=-1), 2, post_process=np.all)


class _Interrupt(Exception):
    """Raised to stop an ARMC run halfway."""


def _interrupt_after(iterable: Any, n: int) -> Generator[Any, None, None]:
    """Yield the first **n** items of **iterable**, then raise :exc:`_Interrupt`."""
    for i, item in enumerate(iterable):
        if i == n:
            raise _Interrupt
        yield item


def _get_seeded_armc(name: str, path: Path, folder: str, seed: Any, n: int) -> Tuple[Any, Any]:
    """Construct a short ARMC run from **name** with the given seed and results hook."""
    with open(PATH / name, 'r') as f:
        dct = yaml.load(f.read(), Loader=UniqueLoader)
    dct['monte_carlo'].update(iter_len=20, path=str(path), folder=folder, seed=seed)
    armc, job_kwargs = dict_to_armc(dct)
    ref = ARMC_REF if name == 'armc_ref.yaml' else ARMCPT_REF
    armc.package_manager.hook = iter(load_results(ref, n=n)[:21])
    return armc, job_kwargs


@pytest.mark.parametrize('name,n', [
    ('armc_ref.yaml', 1),
    ('armcpt_ref.yaml', 3),
], ids=['armc', 'armcpt'])
def test_armc_seed(name: str, n: int, tmp_path: Path) -> None:
    """Test that ARMC runs with the same ``monte_carlo.seed`` make the same moves."""
    for folder, seed in [('run1', 1), ('run2', 1), ('run3', 2)]:
        armc, job_kwargs = _get_seeded_armc(name, tmp_path, folder, seed, n)
        assertion.eq(armc.seed, seed)
        hook, armc.package_manager.hook = armc.package_manager.hook, None
        assertion.eq(armc.to_yaml_dict()['monte_carlo']['seed'], seed)
        armc.package_manager.hook = hook
        run_armc(armc, restart=False, **job_kwargs)

    hdf5 = [tmp_path / folder / 'armc.hdf5' for folder in ('run1', 'run2', 'run3')]
    with h5py.File(hdf5[0], 'r') as f1, h5py.File(hdf5[1], 'r') as f2, \
            h5py.File(hdf5[2], 'r') as f3:
        assertion.eq(f1.attrs['seed'], 1)
        assertion.eq(f3.attrs['seed'], 2)
        np.testing.assert_array_equal(f1['param'][:], f2['param'][:])
        np.testing.assert_array_equal(f1['aux_error_mod'][:], f2['aux_error_mod'][:])
        np.testing.assert_array_equal(f1['phi'][:], f2['phi'][:])
        assertion.assert_(np.array_equal, f1['param'][:], f3['param'][:], invert=True)


@pytest.mark.parametrize('n_results', [8, 13, 19])
def test_armc_restart_seed(n_results: int, tmp_path: Path) -> None:
    """Test that an interrupted and restarted ARMC run makes the same moves as one without."""
    armc1, job_kwargs1 = _get_seeded_armc('armc_ref.yaml', tmp_path, 'ref', 1, n=1)
    run_armc(armc1, restart=False, **job_kwargs1)

    # Stop the second run halfway, then restart it with a different seed:
    # the restart must continue from the generator state stored in the .hdf5 file
    armc2, job_kwargs2 = _get_seeded_armc('armc_ref.yaml', tmp_path, 'restart', 1, n=1)
    armc2.package_manager.hook = _interrupt_after(armc2.package_manager.hook, n_results)
    with pytest.raises(_Interrupt):
        run_armc(armc2, restart=False, **job_kwargs2)

    armc3, job_kwargs3 = _get_seeded_armc('armc_ref.yaml', tmp_path, 'restart', 2, n=1)
    armc3.package_manager.hook = iter(load_results(ARMC_REF, n=1)[n_results:21])
    run_armc(armc3, restart=True, **job_kwargs3)
    assertion.assert_(next, armc3.package_manager.hook, exception=StopIteration)

    with h5py.File(tmp_path / 'ref' / 'armc.hdf5', 'r') as f1, \
            h5py.File(tmp_path / 'restart' / 'armc.hdf5', 'r') as f2:
        assertion.eq(f2.attrs['seed'], 1)
        np.testing.assert_array_equal(f1['param'][:], f2['param'][:])
        np.testing.assert_array_equal(f1['acceptance'][:], f2['acceptance'][:])
        np.testing.assert_array_equal(f1['phi'][:], f2['phi'][:])


def test_armc_settings(tmp_path: Path) -> None:
    """Test that the MD settings always hold the current parameters, also after a rejection."""
    armc, job_kwargs = _get_seeded_armc('armc_ref.yaml', tmp_path, 'run', 1, n=1)
    get_pes_descriptors = armc.get_pes_descriptors
    n_checks = []

    def check_settings(*args: Any, **kwargs: Any) -> Any:
        settings = armc.package_manager['md'][0]['settings']
        for (key, prm, atoms), value in armc.param.param[0].items():
            df = settings[key]
            if prm in df.index and atoms in df.columns:
                assertion.isclose(float(df.at[prm, atoms]), value, message=f"{prm} {atoms}")
        n_checks.append(None)
        return get_pes_descriptors(*args, **kwargs)

    armc.get_pes_descriptors = check_settings  # type: ignore[method-assign]
    run_armc(armc, restart=False, **job_kwargs)
    assertion.eq(len(n_checks), 21)
    with h5py.File(tmp_path / 'run' / 'armc.hdf5', 'r') as f:
        assertion.assert_(np.all, f['acceptance'][:], invert=True)


@pytest.mark.parametrize('seed,exc', [
    (-1, ValueError),
    (2**63, ValueError),
    (True, TypeError),
    (1.5, TypeError),
], ids=['negative', 'too_large', 'bool', 'float'])
def test_armc_seed_raise(seed: Any, exc: type[Exception]) -> None:
    """Test that :class:`ARMC` rejects invalid seeds."""
    with open(PATH / 'armc_ref.yaml', 'r') as f:
        dct = yaml.load(f.read(), Loader=UniqueLoader)
    armc, _ = dict_to_armc(dct)
    kwargs = {k: getattr(armc, k) for k in ('phi', 'param', 'package_manager', 'molecule')}
    with pytest.raises(exc):
        type(armc)(seed=seed, **kwargs)

    dct['monte_carlo']['seed'] = seed
    with pytest.raises(SchemaError):
        dict_to_armc(dct)


def test_hdf5_file_path(tmp_path: Path) -> None:
    """Test that an ``monte_carlo.hdf5_file`` in an existing directory is passed to ARMC."""
    with open(PATH / 'armc_ref.yaml', 'r') as f:
        dct = yaml.load(f.read(), Loader=UniqueLoader)
    hdf5_file = tmp_path / 'results.hdf5'
    dct['monte_carlo']['hdf5_file'] = str(hdf5_file)

    armc, job_kwargs = dict_to_armc(dct)
    assertion.eq(armc.hdf5_file, str(hdf5_file))
    assertion.contains(job_kwargs, 'hdf5_file', invert=True)


def test_param_sorting() -> None:
    """Tests for :func:`FOX.armc.sanitization._sort_atoms`"""
    df1 = DF.copy()
    df1.loc[6, "atoms"] = "Se Cd"
    df1.loc[13, "atoms"] = "Se O_1"
    _sort_atoms(df1)
    np.testing.assert_array_equal(df1, DF)

    df2 = DF.copy()
    df2.loc[4, "atoms"] = "Se Cd"
    df2.loc[5, "atoms"] = "Cd Se"
    df2.loc[6, "atoms"] = "Se Cd"
    assertion.assert_(_sort_atoms, df2, exception=ValueError)


@delete_finally(PATH / '_ARMC', PATH / '_ARMCPT')
@pytest.mark.parametrize('name', ['armc_ref.yaml', 'armcpt_ref.yaml'])
def test_yaml_armc(name: str) -> None:
    """Tests for :meth:`FOX.armc.ARMC.to_yaml_dict`."""
    file = PATH / name
    with open(file, 'r') as f:
        _dct = yaml.load(f.read(), Loader=UniqueLoader)

    armc1, job_kwargs1 = dict_to_armc(_dct)
    dct1 = armc1.to_yaml_dict(
        path=job_kwargs1['path'],
        folder=job_kwargs1['folder'],
        logfile=job_kwargs1['logfile'],
        psf=job_kwargs1['psf'],
    )

    armc2, job_kwargs2 = dict_to_armc(dct1)
    dct2 = armc2.to_yaml_dict(
        path=job_kwargs2['path'],
        folder=job_kwargs2['folder'],
        logfile=job_kwargs2['logfile'],
        psf=job_kwargs2['psf'],
    )

    assertion.eq(armc1, armc2)
    assertion.eq(dct1, dct2)


def _has_cp2k() -> bool:
    try:
        subprocess.run('cp2k.ssmp --version', shell=True, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return False
    else:
        return True


HAS_CP2K = _has_cp2k()


@delete_finally(PATH / '_ARMC', PATH / '_ARMCPT')
@pytest.mark.skipif(not HAS_CP2K, reason="Requires CP2K")
@pytest.mark.parametrize('name,dir_name', [
    ('armc_cp2k.yaml', '_ARMC'),
    ('armcpt_cp2k.yaml', '_ARMCPT'),
], ids=['armc', 'armcpt'])
def test_with_cp2k(name: str, dir_name: str) -> None:
    """Test :class:`ARMC` and :class:`ARMCPT`."""
    with open(f'tests/test_files/{name}', 'r') as f:
        dct = yaml.load(f.read(), Loader=UniqueLoader)
    dct["monte_carlo"]["keep_files"] = True
    armc, job_kwargs = dict_to_armc(dct)

    try:
        run_armc(armc, restart=False, **job_kwargs)
    except Exception as ex:
        err_file = PATH / dir_name / 'md' / 'md.err'
        out_file = PATH / dir_name / 'md' / 'md.out'
        has_err_file = os.path.isfile(err_file)
        has_out_file = os.path.isfile(out_file)

        if not (has_err_file or has_out_file):
            raise

        msg = ""
        if has_err_file:
            with open(err_file, 'r', encoding='utf-8') as f:
                msg += f"{err_file}:\n{f.read()}"
        if out_file:
            with open(out_file, 'r', encoding='utf-8') as f:
                msg += f"{out_file}:\n{f.read()}"

        raise AssertionError(msg) from ex
