"""Tests for various ARMC jobs."""

import warnings
import pytest
import shutil
import subprocess
from os import PathLike
from pathlib import Path
from typing import Tuple, Generator, Any, cast, Dict, List, Optional, Union
from itertools import combinations_with_replacement
from types import SimpleNamespace

import numpy as np
import pandas as pd
import h5py
import yaml
from schema import SchemaError
from assertionlib import assertion

from FOX._nanoutils import delete_finally, UniqueLoader
from FOX.testing_utils import load_results, compare_hdf5
from FOX.armc import dict_to_armc, run_armc, ARMC, StopCriteria
from FOX.armc.armc_pt import swap_random
from FOX.armc.err_funcs import default_error_func
from FOX.armc.sanitization import _sort_atoms
from FOX.armc.schemas import mc_schema
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
@pytest.mark.parametrize('name', [
    'armc_ref.yaml',
    pytest.param('armcpt_ref.yaml', marks=pytest.mark.slow),
])
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


def _get_seeded_armc(
    name: str,
    path: Path,
    folder: str,
    seed: Any,
    n: int,
    *,
    pes: Optional[Dict[str, Any]] = None,
    phi: Optional[Dict[str, Any]] = None,
    **mc_kwargs: Any,
) -> Tuple[Any, Any]:
    """Construct a short ARMC run from **name** with the given seed and results hook.

    **pes** and **phi** are added to the respective blocks of the input,
    **mc_kwargs** to its ``monte_carlo`` block.

    """
    with open(PATH / name, 'r') as f:
        dct = yaml.load(f.read(), Loader=UniqueLoader)
    dct['monte_carlo'].update(iter_len=20, path=str(path), folder=folder, seed=seed)
    dct['monte_carlo'].update(mc_kwargs)
    dct['pes'].update(pes or {})
    dct.setdefault('phi', {}).update(phi or {})
    armc, job_kwargs = dict_to_armc(dct)
    ref = ARMC_REF if name == 'armc_ref.yaml' else ARMCPT_REF
    armc.package_manager.hook = iter(load_results(ref, n=n)[:1 + armc.iter_len])
    return armc, job_kwargs


@pytest.mark.parametrize('name,n', [
    ('armc_ref.yaml', 1),
    pytest.param('armcpt_ref.yaml', 3, marks=pytest.mark.slow),
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


# 19: interrupted right after a rejected move
@pytest.mark.parametrize('n_results', [
    pytest.param(8, marks=pytest.mark.slow),
    pytest.param(13, marks=pytest.mark.slow),
    19,
])
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


#: A second RDF, for tests with more than one PES descriptor
RDF2 = {'rdf2': {'func': 'FOX.MultiMolecule.init_rdf', 'kwargs': {'atom_subset': ['Cd', 'Se']}}}


def _n_finished(hdf5: Union[str, "PathLike[str]"]) -> int:
    """Return the number of finished iterations in an ARMC .hdf5 file."""
    with h5py.File(hdf5, 'r') as f:
        i, j = f.attrs['super-iteration'], f.attrs['sub-iteration']
        return int(1 + j + i * f['param'].shape[1])


def test_get_aux_error() -> None:
    """Test that the errors are grouped by molecule or parameter set, then by PES descriptor."""
    def err_func(ref: Any, value: Any) -> Any:
        return value

    keys = [f'{name}.{i}' for name in ('rdf', 'energy') for i in range(3)]
    pes = {k: SimpleNamespace(ref=None, err_func=err_func, weight=1.0) for k in keys}
    values = {k: 10 * ('energy' in k) + int(k[-1]) for k in keys}

    aux_error = ARMC.get_aux_error(SimpleNamespace(pes=pes), values)  # type: ignore[arg-type]
    np.testing.assert_array_equal(aux_error, [[0, 10], [1, 11], [2, 12]])


def test_swap_random_weighted() -> None:
    """Test :func:`swap_random` with ``weighted=True``."""
    armcpt = SimpleNamespace(
        phi=SimpleNamespace(a_target=np.full(3, 0.25)),
        rng=np.random.default_rng(0),
        logger=SimpleNamespace(info=lambda msg: None),
    )

    # Acceptance rates below and above the target; picking parameter sets used to raise here
    acceptance = np.zeros((100, 3), dtype=np.bool_)
    acceptance[:10, 0] = acceptance[:50, 1] = acceptance[:20, 2] = True
    for _ in range(10):
        for idx in swap_random(acceptance, armcpt, weighted=True):  # type: ignore[arg-type]
            assertion.eq(len(set(idx)), 2)

    # A parameter set right at its target acceptance rate is always picked
    acceptance[:25, 2] = True
    assertion.eq(swap_random(acceptance, armcpt, weighted=True), [])  # type: ignore[arg-type]


@pytest.mark.parametrize('stop,n_iter', [
    ({'target_error': 0.2}, 1),
    ({'patience': 1}, 10),
    ({'max_hours': 1e-9}, 1),
    ({'target_error': 0.1, 'patience': 2}, 20),
    (None, 20),
], ids=['target_error', 'patience', 'max_hours', 'not_met', 'none'])
def test_armc_stop(stop: Optional[Dict[str, Any]], n_iter: int, tmp_path: Path) -> None:
    """Test the ``monte_carlo.stop`` criteria.

    The initial parameters have an error of 0.171 and none of the 20 iterations improves on it.

    """
    armc, job_kwargs = _get_seeded_armc('armc_ref.yaml', tmp_path, 'run', 1, n=1, stop=stop)
    assertion.eq(armc.stop, StopCriteria(**(stop or {})))
    run_armc(armc, restart=False, **job_kwargs)

    hdf5 = tmp_path / 'run' / 'armc.hdf5'
    assertion.eq(_n_finished(hdf5), n_iter)
    with h5py.File(hdf5, 'r') as f:
        assertion.isclose(f.attrs['initial_error'], 0.17074477981934233)
        assertion.eq(f.attrs.get('stop_reason'), armc.stop_reason)
    assertion.eq(armc.stop_reason is None, n_iter == 20)


def test_armc_stop_restart(tmp_path: Path) -> None:
    """Test restarting an ARMC run that stopped early."""
    armc0, job_kwargs0 = _get_seeded_armc('armc_ref.yaml', tmp_path, 'ref', 1, n=1)
    run_armc(armc0, restart=False, **job_kwargs0)

    stop = {'patience': 1}
    armc1, job_kwargs1 = _get_seeded_armc('armc_ref.yaml', tmp_path, 'run', 1, n=1, stop=stop)
    run_armc(armc1, restart=False, **job_kwargs1)
    assertion.eq(_n_finished(tmp_path / 'run' / 'armc.hdf5'), 10)

    # The same criteria: stop again, without running any job
    armc2, job_kwargs2 = _get_seeded_armc('armc_ref.yaml', tmp_path, 'run', 1, n=1, stop=stop)
    armc2.package_manager.hook = iter([])
    run_armc(armc2, restart=True, **job_kwargs2)
    assertion.eq(armc2.stop_reason, armc1.stop_reason)

    # No criteria: continue as if the run never stopped
    armc3, job_kwargs3 = _get_seeded_armc('armc_ref.yaml', tmp_path, 'run', 1, n=1)
    armc3.package_manager.hook = iter(load_results(ARMC_REF, n=1)[11:21])
    run_armc(armc3, restart=True, **job_kwargs3)
    assertion.assert_(next, armc3.package_manager.hook, exception=StopIteration)
    assertion.is_(armc3.stop_reason, None)
    with h5py.File(tmp_path / 'ref' / 'armc.hdf5', 'r') as f1, \
            h5py.File(tmp_path / 'run' / 'armc.hdf5', 'r') as f2:
        assertion.eq(f2.attrs['stop_reason'], '')
        for key in ('param', 'acceptance', 'phi', 'aux_error'):
            np.testing.assert_array_equal(f1[key][:], f2[key][:], err_msg=key)


@pytest.mark.parametrize('dct', [
    {'n_jobs': 0},
    {'n_jobs': True},
    {'stop': {'target_error': -1}},
    {'stop': {'patience': 0}},
    {'stop': {'patience': 1.5}},
    {'stop': {'max_hours': 0}},
    {'stop': {'max_errors': 1}},
    {'stop': [1]},
], ids=['n_jobs_0', 'n_jobs_bool', 'target_error', 'patience_0', 'patience_float',
        'max_hours', 'unknown_key', 'not_a_mapping'])
def test_mc_schema_raise(dct: Dict[str, Any]) -> None:
    """Test that the ``monte_carlo`` schema rejects invalid ``n_jobs`` and ``stop`` values."""
    with pytest.raises(SchemaError):
        mc_schema.validate(dct)


def test_mc_schema_defaults() -> None:
    """Test the defaults of ``monte_carlo.n_jobs`` and ``monte_carlo.stop``."""
    dct = mc_schema.validate({'stop': {'patience': '5'}})
    assertion.eq(dct['n_jobs'], 1)
    assertion.eq(dct['stop'], {'target_error': None, 'patience': 5, 'max_hours': None})
    dct = mc_schema.validate({'stop': None})
    assertion.eq(dct['stop'], {'target_error': None, 'patience': None, 'max_hours': None})


def test_armc_n_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ``monte_carlo.n_jobs`` is passed on to the package manager."""
    armc, job_kwargs = _get_seeded_armc('armc_ref.yaml', tmp_path, 'run', 1, n=1, n_jobs=3)
    assertion.eq(armc.n_jobs, 3)
    hook, armc.package_manager.hook = armc.package_manager.hook, None
    assertion.eq(armc.to_yaml_dict()['monte_carlo']['n_jobs'], 3)
    armc.package_manager.hook = hook

    n_processes = []
    cls = type(armc.package_manager)
    call = cls.__call__

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        n_processes.append(kwargs.get('n_processes'))
        return call(self, *args, **kwargs)

    monkeypatch.setattr(cls, '__call__', wrapper)
    run_armc(armc, restart=False, **job_kwargs)
    assertion.eq(n_processes, 21 * [3])


def test_armc_phi_auto(tmp_path: Path) -> None:
    """Test ``phi.phi = "auto"``."""
    armc, job_kwargs = _get_seeded_armc(
        'armc_ref.yaml', tmp_path, 'run', 1, n=1, pes=RDF2, phi={'phi': 'auto'}, iter_len=10,
    )
    assertion.is_(armc.phi.auto, True)
    hook, armc.package_manager.hook = armc.package_manager.hook, None
    assertion.eq(armc.to_yaml_dict()['phi']['phi'], 'auto')
    armc.package_manager.hook = hook

    run_armc(armc, restart=False, **job_kwargs)
    with h5py.File(tmp_path / 'run' / 'armc.hdf5', 'r') as f:
        # phi is added to both terms of the error (one per PES descriptor)
        np.testing.assert_allclose(f['phi'][0], 0.01 * f.attrs['initial_error'] / 2)


def test_armc_restart_two_pes(tmp_path: Path) -> None:
    """Test that a restart with two PES descriptors continues as if it was never interrupted."""
    armc1, job_kwargs1 = _get_seeded_armc('armc_ref.yaml', tmp_path, 'ref', 1, n=1, pes=RDF2)
    run_armc(armc1, restart=False, **job_kwargs1)

    armc2, job_kwargs2 = _get_seeded_armc('armc_ref.yaml', tmp_path, 'restart', 1, n=1, pes=RDF2)
    armc2.package_manager.hook = _interrupt_after(armc2.package_manager.hook, 13)
    with pytest.raises(_Interrupt):
        run_armc(armc2, restart=False, **job_kwargs2)

    armc3, job_kwargs3 = _get_seeded_armc('armc_ref.yaml', tmp_path, 'restart', 1, n=1, pes=RDF2)
    armc3.package_manager.hook = iter(load_results(ARMC_REF, n=1)[13:21])
    run_armc(armc3, restart=True, **job_kwargs3)

    # The restart used to keep only the error of the first PES descriptor
    assertion.eq({v.shape for v in armc3.values()}, {(1, 2)})
    for k, v in armc3.items():
        np.testing.assert_allclose(v, armc1[k], err_msg=str(k))
    with h5py.File(tmp_path / 'ref' / 'armc.hdf5', 'r') as f1, \
            h5py.File(tmp_path / 'restart' / 'armc.hdf5', 'r') as f2:
        for key in ('param', 'acceptance', 'phi', 'aux_error'):
            np.testing.assert_array_equal(f1[key][:], f2[key][:], err_msg=key)


def test_armcpt_aux_error(tmp_path: Path) -> None:
    """Test that every ARMC-PT parameter set is judged on its own errors."""
    armc, job_kwargs = _get_seeded_armc(
        'armcpt_ref.yaml', tmp_path, 'run', 1, n=3, pes=RDF2, iter_len=4, sub_iter_len=2,
    )
    run_armc(armc, restart=False, **job_kwargs)

    with h5py.File(tmp_path / 'run' / 'armc.hdf5', 'r') as f:
        aux_error = f['aux_error'][:]
        index = f['aux_error'].attrs['index'].astype(str).tolist()
        assertion.eq(index, ['rdf.0', 'rdf.1', 'rdf.2', 'rdf2.0', 'rdf2.1', 'rdf2.2'])
        for j, name in enumerate(['rdf', 'rdf2']):
            for i in range(3):
                ref = f[f'{name}.{i}.ref'][:]
                for kappa, omega in np.ndindex(2, 2):
                    err = default_error_func(ref, f[f'{name}.{i}'][kappa, omega])
                    assertion.isclose(aux_error[kappa, omega, j, i], err, rel_tol=1e-12)


def test_armcpt_restart(tmp_path: Path) -> None:
    """Test that an ARMC-PT restart keeps the swapped settings of the parameter sets."""
    def swap(*args: Any) -> List[Tuple[int, int]]:
        return [(0, 2)]

    kwargs: Dict[str, Any] = dict(n=3, iter_len=6, sub_iter_len=2)
    armc1, job_kwargs1 = _get_seeded_armc('armcpt_ref.yaml', tmp_path, 'ref', 1, **kwargs)
    armc1.swap_phi = swap
    run_armc(armc1, restart=False, **job_kwargs1)

    # Interrupt after the swap at the end of the first super-iteration
    armc2, job_kwargs2 = _get_seeded_armc('armcpt_ref.yaml', tmp_path, 'restart', 1, **kwargs)
    armc2.swap_phi = swap
    armc2.package_manager.hook = _interrupt_after(armc2.package_manager.hook, 4)
    with pytest.raises(_Interrupt):
        run_armc(armc2, restart=False, **job_kwargs2)
    np.testing.assert_array_equal(armc2.phi.a_target, [0.5, 0.375, 0.25])

    armc3, job_kwargs3 = _get_seeded_armc('armcpt_ref.yaml', tmp_path, 'restart', 1, **kwargs)
    armc3.swap_phi = swap
    armc3.package_manager.hook = iter(load_results(ARMCPT_REF, n=3)[4:7])
    run_armc(armc3, restart=True, **job_kwargs3)
    assertion.assert_(next, armc3.package_manager.hook, exception=StopIteration)

    np.testing.assert_array_equal(armc3.phi.a_target, armc1.phi.a_target)
    np.testing.assert_array_equal(armc3.phi.gamma, armc1.phi.gamma)
    np.testing.assert_array_equal(armc3.param.move_range, armc1.param.move_range)
    with h5py.File(tmp_path / 'ref' / 'armc.hdf5', 'r') as f1, \
            h5py.File(tmp_path / 'restart' / 'armc.hdf5', 'r') as f2:
        for key in ('param', 'acceptance', 'phi', 'aux_error'):
            np.testing.assert_array_equal(f1[key][:], f2[key][:], err_msg=key)


@pytest.fixture(scope='module')
def armc_kwargs() -> Tuple[type, Dict[str, Any]]:
    """Return the ARMC type and the arguments for constructing it from armc_ref.yaml."""
    with open(PATH / 'armc_ref.yaml', 'r') as f:
        dct = yaml.load(f.read(), Loader=UniqueLoader)
    armc, _ = dict_to_armc(dct)
    kwargs = {k: getattr(armc, k) for k in ('phi', 'param', 'package_manager', 'molecule')}
    return type(armc), kwargs


@pytest.mark.parametrize('seed,exc', [
    (-1, ValueError),
    (2**63, ValueError),
    (True, TypeError),
    (1.5, TypeError),
], ids=['negative', 'too_large', 'bool', 'float'])
def test_armc_seed_raise(
    seed: Any, exc: type[Exception], armc_kwargs: Tuple[type, Dict[str, Any]]
) -> None:
    """Test that :class:`ARMC` and the ``monte_carlo`` schema reject invalid seeds."""
    cls, kwargs = armc_kwargs
    with pytest.raises(exc):
        cls(seed=seed, **kwargs)
    with pytest.raises(SchemaError):
        mc_schema.validate({'seed': seed})
    assertion.eq(mc_schema.validate({'seed': 5})['seed'], 5)


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
@pytest.mark.parametrize('name', [
    'armc_ref.yaml',
    pytest.param('armcpt_ref.yaml', marks=pytest.mark.slow),
])
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
@pytest.mark.slow
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
        # The MD jobs are named `md.0`, `md.1`, ... if they run in parallel
        files = sorted((PATH / dir_name).glob('md*/md*.err'))
        files += sorted((PATH / dir_name).glob('md*/md*.out'))
        if not files:
            raise

        msg = ""
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                msg += f"{file}:\n{f.read()}"
        raise AssertionError(msg) from ex

    # Every molecule (ARMC) or parameter set (ARMC-PT) must have received its own results
    with h5py.File(PATH / dir_name / 'armc.hdf5', 'r') as f:
        assertion.assert_(np.isfinite(f['aux_error'][:]).all)
