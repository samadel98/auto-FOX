"""Tests for :class:`FOX.armc.CMAES`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Generator, Iterable, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml
from assertionlib import assertion

from FOX import from_hdf5
from FOX._nanoutils import UniqueLoader
from FOX.armc import dict_to_armc, run_armc, CMAES
from FOX.entry_points import main_armc_report
from FOX.recipes import get_best
from FOX.testing_utils import load_results
from FOXdata import ARMCPT_DIR as ARMCPT_REF

PATH = Path('tests') / 'test_files'


def _get_cmaes(
    path: Path, folder: str, seed: int = 1, *, n_results: int = 6, **mc_kwargs: Any,
) -> Tuple[CMAES, Dict[str, Any]]:
    """Construct a short CMA-ES run with 3 candidates per generation.

    The MD jobs are replaced by the (unrelated) results of an ARMC-PT run with 3 parameter sets,
    skipping its first iteration as all three of its parameter sets are identical there.

    """
    with open(PATH / 'armc_ref.yaml', 'r') as f:
        dct = yaml.load(f.read(), Loader=UniqueLoader)
    dct['monte_carlo'].update(
        type='FOX.armc.CMAES', popsize=3, iter_len=6, sub_iter_len=2,
        path=str(path), folder=folder, seed=seed,
    )
    dct['monte_carlo'].update(mc_kwargs)
    armc, job_kwargs = dict_to_armc(dct)
    assert isinstance(armc, CMAES)
    armc.package_manager.hook = iter(load_results(ARMCPT_REF, n=3)[1:1 + n_results])
    return armc, job_kwargs


class _Interrupt(Exception):
    """Raised by :func:`_interrupt_after` to simulate an interrupted calculation."""


def _interrupt_after(iterable: Iterable[Any], n: int) -> Generator[Any, None, None]:
    """Yield the first **n** items of **iterable**, then raise :exc:`_Interrupt`."""
    iterator = iter(iterable)
    for _ in range(n):
        yield next(iterator)
    raise _Interrupt


def _charges(armc: CMAES, param: np.ndarray) -> pd.Series:
    """Return the charges of the parameters **param**, ordered as ``armc.param.param``."""
    is_charge = armc.param.param.index.get_level_values(1) == 'charge'
    index = armc.param.param.index[is_charge].get_level_values(2)
    return pd.Series(param[is_charge], index=index)


def _net_charge(armc: CMAES, param: np.ndarray) -> float:
    """Return the net charge of the parameters **param**, ordered as ``armc.param.param``."""
    count = armc.param.metadata.loc[('charge', 'charge'), (0, 'count')].astype(np.float64)
    return float((_charges(armc, param) * count).sum())


def _ligand_charge(armc: CMAES, param: np.ndarray) -> float:
    """Return the charge of the ligand in the constraint ``Cd == -2 * $LIGAND``."""
    _, ligand = armc.param.constraints['charge', 'charge']
    return float((_charges(armc, param)[ligand.index] * ligand).sum())


def test_cmaes(tmp_path: Path) -> None:
    """Test a short CMA-ES run."""
    armc, job_kwargs = _get_cmaes(tmp_path, 'run')
    param0 = armc.param.param[0].values.copy()
    run_armc(armc, restart=False, **job_kwargs)
    assertion.assert_(next, armc.package_manager.hook, exception=StopIteration)
    assertion.eq(armc.n_candidates, 3)
    assertion.is_(armc.stop_reason, None)

    with h5py.File(tmp_path / 'run' / 'armc.hdf5', 'r') as f:
        param = f['param'][:].reshape(-1, 3, len(param0))
        aux_error = f['aux_error'][:].reshape(-1, 3)
        acceptance = f['acceptance'][:].reshape(-1, 3)
        phi = f['phi'][:]
        initial_error = f.attrs['initial_error']
        assertion.gt(f['cmaes_state'].size, 0)

    # The first candidate holds the initial parameters, with their charges moved
    # by < 1e-5 to satisfy `Cd == -2 * $LIGAND` and the (integer) net charge exactly
    np.testing.assert_allclose(param[0, 0], param0, rtol=1e-4)
    assertion.isclose(initial_error, aux_error[0, 0])

    # Every generation selects the best candidate (mu = 1 for 3 candidates)
    np.testing.assert_array_equal(acceptance.sum(axis=1), 1)
    np.testing.assert_array_equal(acceptance.argmax(axis=1), aux_error.argmin(axis=1))
    assertion.assert_(np.all, (phi > 0) & (phi < 0.1))

    # Frozen parameters stay put, bounds and the net charge are respected
    frozen = armc.param.metadata[0, 'frozen'].values.astype(bool)
    np.testing.assert_array_equal(
        param[..., frozen], np.broadcast_to(param0[frozen], param[..., frozen].shape)
    )
    prm_min = armc.param.metadata[0, 'min'].values.astype(np.float64)
    prm_max = armc.param.metadata[0, 'max'].values.astype(np.float64)
    assertion.assert_(np.all, (prm_min <= param) & (param <= prm_max))
    assertion.eq(armc.param._net_charge[0], 0)
    for p in param.reshape(-1, len(param0)):
        assertion.isclose(_net_charge(armc, p), 0, abs_tol=1e-8)
        assertion.isclose(_charges(armc, p)['Cd'], _ligand_charge(armc, p), abs_tol=1e-8)


def test_cmaes_seed(tmp_path: Path) -> None:
    """Test that CMA-ES runs with the same seed draw the same candidates."""
    params = []
    for folder, seed in [('run1', 1), ('run2', 1), ('run3', 2)]:
        armc, job_kwargs = _get_cmaes(tmp_path, folder, seed)
        run_armc(armc, restart=False, **job_kwargs)
        with h5py.File(tmp_path / folder / 'armc.hdf5', 'r') as f:
            params.append(f['param'][:])
    np.testing.assert_array_equal(params[0], params[1])
    assertion.assert_(np.array_equal, params[0], params[2], invert=True)


@pytest.mark.parametrize('n_results', [1, 3, 4])
def test_cmaes_restart(n_results: int, tmp_path: Path) -> None:
    """Test that a restarted CMA-ES run draws the same candidates as an uninterrupted one."""
    armc1, job_kwargs1 = _get_cmaes(tmp_path, 'ref')
    run_armc(armc1, restart=False, **job_kwargs1)

    armc2, job_kwargs2 = _get_cmaes(tmp_path, 'restart')
    armc2.package_manager.hook = _interrupt_after(armc2.package_manager.hook, n_results)
    with pytest.raises(_Interrupt):
        run_armc(armc2, restart=False, **job_kwargs2)

    # A different seed: the restart continues from the state stored in the .hdf5 file
    armc3, job_kwargs3 = _get_cmaes(tmp_path, 'restart', seed=2)
    armc3.package_manager.hook = iter(load_results(ARMCPT_REF, n=3)[1 + n_results:7])
    run_armc(armc3, restart=True, **job_kwargs3)
    assertion.assert_(next, armc3.package_manager.hook, exception=StopIteration)

    with h5py.File(tmp_path / 'ref' / 'armc.hdf5', 'r') as f1, \
            h5py.File(tmp_path / 'restart' / 'armc.hdf5', 'r') as f2:
        for key in ('param', 'acceptance', 'phi', 'aux_error'):
            np.testing.assert_array_equal(f1[key][:], f2[key][:], err_msg=key)
        assertion.eq(f1.attrs['initial_error'], f2.attrs['initial_error'])


def _use_quadratic_error(armc: CMAES, factor: np.ndarray) -> None:
    """Replace the MD jobs and PES descriptors of **armc** by a quadratic function.

    The minimum lies at the initial parameters times **factor**, except for the
    Se and OG2D2 charges, which follow from the net charge and ``Cd == -2 * $LIGAND``.

    """
    idx = armc._variable_index()
    p0 = armc.param.param.loc[idx, 0].values.astype(np.float64)
    target = p0 * factor

    is_charge = idx.get_level_values(1) == 'charge'
    if is_charge.any():
        q = _charges(armc, armc.param.param[0].values.astype(np.float64))
        q_var = pd.Series(target[is_charge], index=idx[is_charge].get_level_values(2))
        q[q_var.index] = q_var
        count = armc.param.metadata.loc[('charge', 'charge'), (0, 'count')].astype(np.float64)
        q['OG2D2'] = (-q['Cd'] - 2 * q['CG2O3'] - 2 * q['HGR52']) / 4
        q['Se'] = 0.0
        q['Se'] = -(q * count).sum() / count['Se']
        target[is_charge] = q[q_var.index].values

    def err_func(ref: np.ndarray, value: np.ndarray) -> float:
        return float(np.sum((value - ref)**2))

    armc.pes = {'x.0': SimpleNamespace(ref=np.zeros(len(idx)), err_func=err_func, weight=1.0)}
    armc.pes_validation = {}

    def get_pes_descriptors(
        get_first_key: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], None]:
        param = armc.param.param.loc[idx]
        ret = {f'x.{i}': (param[i].values - target) / np.abs(p0) for i in param.columns}
        return ret, {}, None

    armc.get_pes_descriptors = get_pes_descriptors  # type: ignore[method-assign]


@pytest.mark.parametrize('stop', [None, {'target_error': 1e-3}], ids=['iter_len', 'target_error'])
def test_cmaes_converges(stop: Optional[Dict[str, Any]], tmp_path: Path) -> None:
    """Test that the CMA-ES finds the minimum of a quadratic function."""
    armc, job_kwargs = _get_cmaes(tmp_path, 'run', popsize=8, iter_len=100, sub_iter_len=10,
                                  stop=stop)
    rng = np.random.default_rng(0)
    _use_quadratic_error(armc, rng.uniform(0.85, 1.15, size=len(armc._variable_index())))
    run_armc(armc, restart=False, **job_kwargs)

    with h5py.File(tmp_path / 'run' / 'armc.hdf5', 'r') as f:
        initial_error = f.attrs['initial_error']
        error = f['aux_error'][:].reshape(-1, 8)
    best_error = np.nanmin(error)
    if stop is None:
        assertion.is_(armc.stop_reason, None)
        assertion.lt(best_error, 1e-4 * initial_error)
    else:
        assertion.contains(armc.stop_reason, 'target_error')
        assertion.le(best_error, 1e-3)
        assertion.lt(np.isfinite(error).sum(), 8 * 100)


@pytest.mark.parametrize('dct,exc', [
    ({'type': 'FOX.armc.ARMC', 'popsize': 4}, ValueError),
    ({'type': 'FOX.armc.ARMC', 'sigma': 0.1}, ValueError),
    ({'popsize': 1}, Exception),
    ({'popsize': True}, Exception),
    ({'sigma': 0}, Exception),
], ids=['armc_popsize', 'armc_sigma', 'popsize_1', 'popsize_bool', 'sigma_0'])
def test_cmaes_input_raise(dct: Dict[str, Any], exc: type, tmp_path: Path) -> None:
    """Test that invalid CMA-ES options are rejected."""
    with open(PATH / 'armc_ref.yaml', 'r') as f:
        inp = yaml.load(f.read(), Loader=UniqueLoader)
    inp['monte_carlo'].update(type='FOX.armc.CMAES', path=str(tmp_path))
    inp['monte_carlo'].update(dct)
    with pytest.raises(exc):
        dict_to_armc(inp)


def test_cmaes_phi_raise(tmp_path: Path) -> None:
    """Test that the CMA-ES rejects more than one ``phi`` value."""
    with open(PATH / 'armc_ref.yaml', 'r') as f:
        inp = yaml.load(f.read(), Loader=UniqueLoader)
    inp['monte_carlo'].update(type='FOX.armc.CMAES', path=str(tmp_path))
    inp['phi'] = {'phi': [1, 2]}
    with pytest.raises(ValueError):
        dict_to_armc(inp)


def test_cmaes_popsize_default(tmp_path: Path) -> None:
    """Test the default number of candidates: 4 + 3 ln(n) for n variable parameters."""
    with open(PATH / 'armc_ref.yaml', 'r') as f:
        inp = yaml.load(f.read(), Loader=UniqueLoader)
    inp['monte_carlo'].update(type='FOX.armc.CMAES', path=str(tmp_path))
    armc, _ = dict_to_armc(inp)
    assert isinstance(armc, CMAES)
    n = len(armc._variable_index())
    assertion.eq(n, 14)
    assertion.eq(armc.n_candidates, 4 + int(3 * np.log(n)))


def _without_floats(obj: Any) -> Any:
    """Replace all floats in the nested dictionaries and lists of **obj** with ``0.0``."""
    if isinstance(obj, dict):
        return {k: _without_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_without_floats(v) for v in obj]
    return 0.0 if isinstance(obj, float) else obj


def test_cmaes_yaml(tmp_path: Path) -> None:
    """Test that :meth:`CMAES.to_yaml_dict` gives the same input before and after a run.

    Only the parameter values differ: they are those of the latest first candidate, as
    :class:`ARMC` gives its latest parameters.

    """
    armc, job_kwargs = _get_cmaes(tmp_path, 'run', sigma=0.05)
    hook, armc.package_manager.hook = armc.package_manager.hook, None
    dct1 = armc.to_yaml_dict()
    assertion.eq(dct1['monte_carlo']['popsize'], 3)
    assertion.eq(dct1['monte_carlo']['sigma'], 0.05)

    armc.package_manager.hook = hook
    run_armc(armc, restart=False, **job_kwargs)
    armc.package_manager.hook = None
    dct2 = armc.to_yaml_dict()
    assertion.eq(yaml.dump(_without_floats(dct1)), yaml.dump(_without_floats(dct2)))
    assertion.eq(yaml.dump({**dct1, 'param': None}), yaml.dump({**dct2, 'param': None}))

    # And the output can be read again
    armc2, _ = dict_to_armc(dct2)
    assert isinstance(armc2, CMAES)
    assertion.eq((armc2.popsize, armc2.sigma), (3, 0.05))


def test_cmaes_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test :func:`FOX.entry_points.main_armc_report` with the .hdf5 file of a CMA-ES run."""
    armc, job_kwargs = _get_cmaes(tmp_path, 'run')
    run_armc(armc, restart=False, **job_kwargs)
    hdf5 = tmp_path / 'run' / 'armc.hdf5'

    with h5py.File(hdf5, 'r') as f:
        param = f['param'][:].reshape(-1, 3, f['param'].shape[-1])
        error = f['aux_error'][:].reshape(-1, 3)
    j, i = np.unravel_index(np.argmin(error), error.shape)

    main_armc_report([str(hdf5), '-o', str(tmp_path / 'report'), '--no-plots'])
    out = capsys.readouterr().out
    assertion.contains(out, f"Lowest error: {error[j, i]:.6g} "
                            f"(super-iteration {j // 2}, sub-iteration {j % 2}, "
                            f"parameter set {i})")
    best = pd.read_csv(tmp_path / 'report' / 'best_param.csv', index_col=[0, 1, 2]).squeeze()
    np.testing.assert_allclose(best.values, param[j, i])
    np.testing.assert_allclose(get_best(hdf5, 'param', i=int(i)).values, param[j, i])

    # The internal state of the CMA-ES is left out when reading all datasets
    dct = from_hdf5(hdf5)
    assertion.assert_(next, (k for k in dct if k.startswith('cmaes')), exception=StopIteration)
    assertion.contains(dct, 'rdf.0')
