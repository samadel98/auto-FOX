"""A module for performing Addaptive Rate Monte Carlo (ARMC) forcefield parameter optimizations.

Index
-----
.. currentmodule:: FOX.armc
.. autosummary::
    ARMC

API
---
.. autoclass:: ARMC
    :members:

"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from itertools import repeat
from typing import (
    Tuple, TYPE_CHECKING, Any, Optional, Iterable, Mapping, Callable,
    overload, Dict, List, ClassVar, Collection,
    Literal, cast,
)

import h5py
import numpy as np

from .monte_carlo import MonteCarloABC
from ..classes.multi_mol import MultiMolecule
from ..type_hints import ArrayOrScalar
from ..io.hdf5_utils import (
    create_hdf5, to_hdf5, create_xyz_hdf5, _get_filename_xyz, hdf5_clear_status,
    hdf5_availability,
)

if TYPE_CHECKING:
    from .phi_updater import PhiUpdater
    from ..io.read_psf import PSFContainer
else:
    from ..type_alias import PhiUpdater, PSFContainer

__all__ = ['ARMC']

PesDict = Dict[str, ArrayOrScalar]
PesMapping = Mapping[str, ArrayOrScalar]

MolList = List[MultiMolecule]
MolCollection = Collection[MultiMolecule]

Key = Tuple[float, ...]


class ARMC(MonteCarloABC):
    r"""The Addaptive Rate Monte Carlo class (:class:`.ARMC`).

    A subclass of :class:`MonteCarloABC`.

    Attributes
    ----------
    iter_len : :class:`int`
        The total number of ARMC iterations :math:`\kappa \omega`.
    super_iter_len : :class:`int`
        The length of each ARMC subiteration :math:`\kappa`.
    sub_iter_len : :class:`int`
        The length of each ARMC subiteration :math:`\omega`.
    phi : :class:`PhiUpdaterABC`
        A PhiUpdater instance.
    \**kwargs : :data:`~typing.Any`
        Keyword arguments for the :class:`MonteCarlo` superclass.

    """

    iter_len: int
    sub_iter_len: int
    phi: PhiUpdater
    HAS_LOOP: ClassVar[Literal[False]] = False

    #: With ``phi.phi = "auto"``, the fraction of the initial error that :math:`\phi`
    #: adds to the error of the current parameters in every iteration.
    AUTO_PHI_FRACTION: ClassVar[float] = 0.01

    @property
    def super_iter_len(self) -> int:
        return self.iter_len // self.sub_iter_len

    def acceptance(self) -> np.ndarray:
        """Create an empty 1D boolean array for holding the acceptance."""
        return np.zeros(self.sub_iter_len, dtype=bool)

    def __init__(self, phi: PhiUpdater,
                 iter_len: int = 50000,
                 sub_iter_len: int = 100,
                 **kwargs: Any) -> None:
        r"""Initialize an :class:`ARMC` instance.

        Parameters
        ----------
        iter_len : :class:`int`
            The total number of ARMC iterations :math:`\kappa \omega`.
        sub_iter_len : :class:`int`
            The length of each ARMC subiteration :math:`\omega`.
        phi : :class:`PhiUpdaterABC`
            A PhiUpdater instance.
        \**kwargs : :data:`~typing.Any`
            Keyword arguments for the :class:`MonteCarlo` superclass.

        """
        super().__init__(**kwargs)

        # Settings specific to addaptive rate Monte Carlo (ARMC)
        self.phi = phi
        if len(phi.phi) != len(self.param.move_range):
            raise ValueError("'phi.phi' and 'param_mapping.move_range' "
                             "should be of the same length")
        self.iter_len = iter_len
        self.sub_iter_len = sub_iter_len

    def to_yaml_dict(  # type: ignore[override]
        self, *,
        path: str = '.',
        folder: str = 'MM_MD_workdir',
        logfile: str = 'armc.log',
        psf: Optional[Iterable[PSFContainer]] = None,
    ) -> Dict[str, Any]:
        """Convert an :class:`ARMC` instance into a .yaml readable by :class:`ARMC.from_yaml`.

        Returns
        -------
        :data:`dict[str, Any] <dict>`
            A dictionary.

        """
        cls = type(self)
        ret: Dict[str, Any] = {}  # type: ignore[typeddict-item]
        ret['param'] = self.param.to_yaml_dict()
        ret['phi'] = self.phi.to_yaml_dict()
        ret['job'] = self.package_manager.to_yaml_dict()
        ret['job']['molecule'] = [m.properties.filename for m in self.molecule]

        if self.molecule[0].lattice is not None:
            ret['job']['lattice'] = lat_lst = []
            iterator1 = (
                (m, MultiMolecule(cast(np.ndarray, m.lattice), atoms={"Xx": range(3)}))
                for m in self.molecule
            )
            for m, lat_mol in iterator1:
                filename = m.properties.filename.replace(".xyz", ".lattice.xyz")
                lat_mol.as_xyz(filename)
                lat_lst.append(filename)
        else:
            ret['job']['lattice'] = None

        ret['monte_carlo'] = {
            'type': f'{cls.__module__}.{cls.__name__}',
            'iter_len': self.iter_len,
            'sub_iter_len': self.sub_iter_len,
            'keep_files': self.keep_files,
            'path': path,
            'folder': folder,
            'logfile': logfile,
            'hdf5_file': self.hdf5_file,
            'seed': self.seed,
            'n_jobs': self.n_jobs,
            'stop': self.stop.to_yaml_dict(),
        }

        # Parse the `pes` and `pes_validation` blocks
        iterator2 = ((k, getattr(self, k)) for k in ('pes', 'pes_validation'))
        for name, attr in iterator2:
            pes: Dict[str, Any] = {}
            ret[name] = pes
            i_max = 1 + max((int(k.split('.')[-1]) for k in attr.keys()), default=0)
            for _k, v in attr.items():
                k, _i = _k.split('.')
                i = int(_i)
                try:
                    pes[k]['kwargs'][i] = v.keywords
                    pes[k]['weight'].append(v.weight)
                except KeyError:
                    pes[k] = {
                        'func': f'{v.args[0].__module__}.{v.args[0].__qualname__}',
                        'kwargs': i_max * [None],
                        'ref': None if v.use_mol else [],
                        'err_func': f'{v.err_func.__module__}.{v.err_func.__qualname__}',
                        'weight': [v.weight],
                    }
                    pes[k]['kwargs'][i] = v.keywords
                if not v.use_mol:
                    pes[k]['ref'].append(v.ref)

        # Parse the `psf` block
        if psf is None:
            ret['psf'] = {'psf_file': None}
        else:
            workdir = Path(path, folder)
            if not os.path.isdir(workdir):
                os.mkdir(workdir)

            ret['psf'] = {'psf_file': []}
            psf_lst = ret['psf']['psf_file']
            for i, psf_obj in enumerate(psf):
                name = str(workdir / f'mol.{i}.psf')
                psf_lst.append(name)
                psf_obj.write(name)
        return ret

    @overload  # type: ignore[override]
    def __call__(self, *, start: None = ..., key_new: None = ...) -> None: ...
    @overload
    def __call__(self, *, start: int, key_new: Key) -> None: ...
    def __call__(self, *, start=None, key_new=None):  # noqa: E301
        """Initialize the Addaptive Rate Monte Carlo procedure."""
        key_new = self._parse_call(start, key_new)
        start_ = start if start is not None else 0
        self._main_loop(start_, key_new)

    def _main_loop(self, start: int, key_new: Any) -> None:
        """Run the super-iterations from **start** onwards, or until a stop criterion is met."""
        for kappa in range(start, self.super_iter_len):
            acceptance = self.acceptance()
            create_xyz_hdf5(self.hdf5_file, self.molecule,
                            iter_len=self.sub_iter_len,
                            phi=self.phi.phi)

            for omega in range(self.sub_iter_len):
                key_new = self.do_inner(kappa, omega, acceptance, key_new)
                if self._stop_after_iteration(kappa, omega):
                    return
            self._end_super_iter(acceptance)
            if self._stop_after_super_iteration(kappa):
                return

    def _end_super_iter(self, acceptance: np.ndarray) -> None:
        r"""Update :math:`\phi` at the end of a super-iteration."""
        self.phi.update(acceptance)

    # Stopping early

    def _error_totals(self, aux_error: np.ndarray) -> np.ndarray:
        """Return the total error of each parameter set in **aux_error**."""
        return np.array([np.sum(aux_error)])

    def _update_best_error(self, aux_error: np.ndarray, kappa: int) -> None:
        """Keep track of the lowest error found so far and the super-iteration it was found in."""
        error = np.min(self._error_totals(aux_error))
        if error < self._best_error:
            self._best_error = float(error)
            self._best_kappa = kappa

    def _stop_after_iteration(self, kappa: int, omega: int) -> bool:
        """Check if the optimization should stop after iteration (**kappa**, **omega**)."""
        if (kappa, omega) == (self.super_iter_len - 1, self.sub_iter_len - 1):
            return False  # The run is over anyway; this is not an early stop
        hours = (time.monotonic() - self._start_time) / 3600
        return self._set_stop_reason(self.stop.check_iteration(self._best_error, hours))

    def _stop_after_super_iteration(self, kappa: int) -> bool:
        """Check if the optimization should stop after super-iteration **kappa**."""
        if kappa == self.super_iter_len - 1:
            return False
        reason = self.stop.check_super_iteration(kappa - self._best_kappa)
        return self._set_stop_reason(reason)

    def _set_stop_reason(self, reason: Optional[str]) -> bool:
        """Log **reason** and store it in the .hdf5 file; return whether there is a reason."""
        if reason is None:
            return False
        self.stop_reason = reason
        self.logger.info(f"Stopping early: {reason}")
        self._set_hdf5_attrs(stop_reason=reason)
        return True

    def _set_hdf5_attrs(self, **attrs: Any) -> None:
        """Set attributes of the .hdf5 file."""
        hdf5_availability(self.hdf5_file)
        with h5py.File(self.hdf5_file, 'r+', libver='latest') as f:
            for k, v in attrs.items():
                f.attrs[k] = v

    @overload
    def _parse_call(self, start: None = ..., key_new: None = ...) -> Key: ...
    @overload
    def _parse_call(self, start: int, key_new: Key) -> Key: ...
    def _parse_call(self, start=None, key_new=None):  # noqa: E301
        """Parse the arguments of :meth:`__call__` and prepare the first key."""
        if start is None:
            create_hdf5(self.hdf5_file, self)  # Construct the HDF5 file

            key_new = self._get_first_key()  # Initialize the first MD calculation
            if np.inf in self[key_new]:
                raise RuntimeError('One or more jobs crashed in the first ARMC iteration; '
                                   'manual inspection of the cp2k output is recomended')
            elif not self.keep_files:
                self.clear_jobs()

            # The error of the initial parameters, the reference for the stop criteria
            self._update_best_error(self[key_new], kappa=-1)
            self._set_hdf5_attrs(initial_error=self._best_error)
            if getattr(self.phi, 'auto', False):
                self._set_auto_phi(self[key_new])

        elif key_new is None:
            raise TypeError("'key_new' cannot be None if 'start' is None")
        return key_new

    def do_inner(self, kappa: int, omega: int, acceptance: np.ndarray, key_old: Key) -> Key:
        r"""Run the inner loop of the :meth:`ARMC.__call__` method.

        Parameters
        ----------
        kappa : :class:`int`
            The super-iteration, :math:`\kappa`, in :meth:`ARMC.__call__`.
        omega : :class:`int`
            The sub-iteration, :math:`\omega`, in :meth:`ARMC.__call__`.
        acceptance : :class:`np.ndarray[np.bool_] <numpy.ndarray>`
            An array with the acceptance over the course of the latest super-iteration
        key_new : :class:`tuple[float, ...]`
            A tuple with the latest set of forcefield parameters.

        Returns
        -------
        :class:`tuple[float, ...]`
            The latest set of parameters.

        """
        # Step 1: Perform a random move
        _key_new = self._do_inner1(key_old, idx=None)

        # Step 2: Calculate PES descriptors
        pes_new, pes_validation, mol_list = self._do_inner2()

        # Step 3: Evaluate the auxiliary error; accept if the new parameter set lowers the error
        error_change, aux_new, aux_validation = self._do_inner3(pes_new, pes_validation, key_old)
        accept = error_change < 0
        self._update_best_error(aux_new, kappa)

        # Step 4: Update the auxiliary error history, apply phi & update job settings
        acceptance[omega] = accept
        key_new = self._do_inner4(accept, error_change, aux_new,
                                  _key_new, key_old, kappa, omega)

        # Step 5: Export the results to HDF5
        self._do_inner5(mol_list, accept, aux_new, aux_validation,
                        pes_new, pes_validation, kappa, omega)
        return key_new

    def _do_inner1(self, key_old: Key, idx: None | int) -> Key:
        """Perform a random move."""
        key_new = self.move(idx=idx)

        if isinstance(key_new, Exception):
            self.logger.warning(f"Recalculating move; {key_new}")
            return self._do_inner1(key_old, idx=idx)

        elif key_new in self:
            # Undo the move first, so the new move is made from the current parameters
            # rather than on top of the already visited ones
            self.logger.info("Recalculating move; move has already been visited")
            self._undo_move(idx)
            return self._do_inner1(key_old, idx=idx)

        return key_new

    def _undo_move(self, idx: None | int) -> None:
        """Reset the parameters moved by :meth:`move` to their last accepted values."""
        if self.param.is_independent:
            i = idx or 0
            self.param.param[i] = self.param.param_old[i]
        else:
            self.param.param[:] = self.param.param_old

    def _do_inner2(self) -> Tuple[PesDict, PesDict, Optional[MolList]]:
        """Calculate PES-descriptors."""
        return self.get_pes_descriptors()

    def _do_inner3(self, pes_new: PesMapping, pes_validation: PesMapping,
                   key_old: Key) -> Tuple[np.float64, np.ndarray, np.ndarray]:
        """Evaluate the auxiliary error; accept if the new parameter set lowers the error."""
        self.logger.info("Calculating auxiliary error")

        aux_new = self.get_aux_error(pes_new)
        aux_validation = self.get_aux_error(pes_validation, validation=True)
        aux_old = self[key_old]
        error_change = (aux_new - aux_old).sum()
        return error_change, aux_new, aux_validation

    def _do_inner4(self, accept: bool | np.bool_, error_change: float, aux_new: np.ndarray,
                   key_new: Key, key_old: Key, kappa: int, omega: int) -> Key:
        """Update the auxiliary error history, apply phi & update job settings."""
        err_round = round(error_change, 4)
        aux_round = round(aux_new.sum(), 4)
        epilog = f'error_change = {err_round}; error = {aux_round}'

        if accept:
            self.logger.info(f"Accepting move {(kappa, omega)}: {epilog}")
            self[key_new] = self.apply_phi(aux_new)
            self.param.param_old[:] = self.param.param
            return key_new
        else:
            self.logger.info(f"Rejecting move {(kappa, omega)}: {epilog}")
            self[key_new] = aux_new
            self[key_old] = self.apply_phi(self[key_old])
            return key_old

    def _do_inner5(self, mol_list: Optional[MolCollection], accept: np.bool_ | np.ndarray,
                   aux_new: np.ndarray, aux_validation: np.ndarray,
                   pes_new: PesMapping, pes_validation: PesMapping,
                   kappa: int, omega: int) -> None:
        """Export the results to HDF5."""
        self.to_hdf5(mol_list, accept, aux_new, aux_validation,
                     pes_new, pes_validation, kappa, omega)

        not_accept = ~np.full_like(self.param.param_old.columns, accept, dtype=np.bool_)
        self.param.param.loc[:, not_accept] = self.param.param_old.loc[:, not_accept]

    @property
    def apply_phi(self) -> Callable[..., np.ndarray]:
        """Apply :attr:`phi` to **value**."""
        return self.phi

    def _set_auto_phi(self, aux_error: np.ndarray) -> None:
        r"""Set :math:`\phi` from the error of the initial parameters, **aux_error**.

        :math:`\phi` is added to every term of the error, so it is divided by the number of terms.

        """
        totals = self._error_totals(aux_error)
        error = np.min(totals)
        n_terms = aux_error.size // len(totals)
        phi = self.AUTO_PHI_FRACTION * error / n_terms
        self.logger.info(f"Setting phi to {phi:.6g}: {100 * self.AUTO_PHI_FRACTION:g}% of "
                         f"the initial error ({error:.6g}) spread over its {n_terms} term(s)")
        self.phi.phi = np.full_like(self.phi.phi, phi)

    def _get_first_key(self, idx: int = 0) -> Key:
        """Create a the ``history_dict`` variable and its first key.

        The to-be returned key servers as the starting argument for :meth:`.do_inner`,
        the latter method relying on both producing and requiring a key as argument.

        Parameters
        ----------
        idx : :class:`int`
            The column key for :attr:`param_mapping["param"]<ARMC.param_mapping.>`.

        Returns
        -------
        :class:`tuple[float, ...]`
            A tuple of Numpy arrays.

        """
        key: Key = tuple(self.param.param[idx].values)
        pes, _, _ = self.get_pes_descriptors(get_first_key=True)

        self[key] = self.get_aux_error(pes)
        self.param.param_old[idx] = self.param.param[idx]
        return key

    def to_hdf5(self, mol_list: Optional[MolCollection],
                accept: np.bool_ | np.ndarray, aux_new: np.ndarray, aux_validation: np.ndarray,
                pes_new: PesMapping, pes_validation: PesMapping,
                kappa: int, omega: int) -> None:
        r"""Construct a dictionary with the **hdf5_kwarg** and pass it to :func:`.to_hdf5`.

        Parameters
        ----------
        mol_list : :class:`list[FOX.MultiMolecule] <list>`, optional
            An iterable consisting molecules instances (or ``None``).
        accept : :class:`bool`
            Whether or not the latest set of parameters was accepted.
        aux_new : :class:`np.ndarray[np.float64] <numpy.ndarray>`
            The latest auxiliary error.
        aux_validation : :class:`np.ndarray[np.float64] <numpy.ndarray>`
            The latest auxiliary error from the validation.
        pes_new : :class:`dict[str, np.ndarray[np.float64]] <dict>`
            A dictionary of PES descriptors.
        pes_validation : :class:`dict[str, np.ndarray[np.float64]] <dict>`
            A dictionary of PES descriptors from the validation.
        kappa : :class:`int`
            The super-iteration, :math:`\kappa`, in :meth:`ARMC.__call__`.
        omega : :class:`int`
            The sub-iteration, :math:`\omega`, in :meth:`ARMC.__call__`.

        Returns
        -------
        :class:`dict[str, Any] <dict>`
            A dictionary with the **hdf5_kwarg** argument for :func:`.to_hdf5`.

        """
        self.logger.info(f"Exporting results to {os.path.basename(self.hdf5_file)!r}\n")

        phi = self.phi.phi
        iterator: Iterable[str]
        if accept.size == 1:
            iterator = repeat(('param' if accept else 'param_old'), len(self.param.param.columns))
        else:
            iterator = ('param' if acc else 'param_old' for acc in np.asarray(accept))
        _aux_error_mod = [getattr(self.param, n)[i].values for i, n in enumerate(iterator)]
        aux_error_mod = np.empty((len(self.param.param.columns), 1 + len(_aux_error_mod[0])))
        aux_error_mod[..., :-1] = _aux_error_mod
        aux_error_mod[..., -1] = phi

        # The auxiliary errors are stored in the order of the PES descriptors,
        # then the molecules (ARMC) or parameter sets (ARMC-PT)
        hdf5_kwarg = {
            'param': self.param.param.values.T,
            'xyz': mol_list,
            'phi': phi,
            'acceptance': accept,
            'aux_error': np.asarray(aux_new).T,
            'aux_error_mod': aux_error_mod,
            'validation/aux_error': np.asarray(aux_validation).T,
        }
        hdf5_kwarg.update(pes_new)
        for k, v in pes_validation.items():
            hdf5_kwarg[f'validation/{k}'] = v

        # Store the state of the random number generator so a restart continues the same moves,
        # and the settings that ARMC-PT swaps between parameter sets
        attrs = {
            'rng_state': json.dumps(self.rng.bit_generator.state),
            'a_target': self.phi.a_target,
            'gamma': self.phi.gamma,
            'move_range': self.param.move_range,
        }
        to_hdf5(self.hdf5_file, hdf5_kwarg, kappa, omega, attrs=attrs)

    def get_aux_error(self, pes_dict: PesMapping, validation: bool = False) -> np.ndarray:
        r"""Return the auxiliary error :math:`\Delta \varepsilon_{QM-MM}`.

        The auxiliary error is constructed using the PES descriptors in **values**
        with respect to **self.ref**.

        The default function is equivalent to:

        .. math::

            \Delta \varepsilon_{QM-MM} =
                \frac{ \sum_{i}^{N} |r_{i}^{QM} - r_{i}^{MM}|^2 }
                {r_{i}^{QM}}

        Parameters
        ----------
        pes_dict : :class:`dict[str, np.ndarray[np.float64]] <dict>`
            An dictionary with :math:`m*n` PES descriptors each.

        Returns
        -------
        :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(m, n)`
            An array with the auxilary errors of :math:`m` molecules (:class:`ARMC`) or
            parameter sets (:class:`ARMCPT`) and :math:`n` PES descriptors.

        """
        # The PES descriptors are `functools.partial` objects with extra attributes
        pes_dict_ref: Mapping[str, Any] = self.pes if not validation else self.pes_validation

        # The keys are of the form `f"{name}.{i}"`,
        # `i` being the index of the molecule or parameter set
        names: Dict[str, int] = {}
        index: List[Tuple[int, int]] = []
        for k in pes_dict.keys():
            name, _, i_str = k.rpartition('.')
            index.append((int(i_str), names.setdefault(name, len(names))))

        length = 1 + max((i for i, _ in index), default=0)
        ret = np.full((length, len(names)), np.nan)
        for (i, j), (k, v) in zip(index, pes_dict.items()):
            ret[i, j] = pes_dict_ref[k].err_func(pes_dict_ref[k].ref, v) * pes_dict_ref[k].weight
        return ret

    def restart(self) -> None:  # type: ignore[override]
        r"""Restart a previously started Addaptive Rate Monte Carlo procedure."""
        # Validate the xyz .hdf5 file; create a new one if required
        xyz = _get_filename_xyz(self.hdf5_file)
        if not os.path.isfile(xyz):
            create_xyz_hdf5(self.hdf5_file, self.molecule, self.sub_iter_len, self.phi.phi)

        # Check that both .hdf5 files can be opened; clear their status if not
        closed1 = hdf5_clear_status(xyz)
        if closed1 is not None:
            self.logger.warning(f"Unable to open {xyz!r}, file status was forcibly reset")

        closed2 = hdf5_clear_status(self.hdf5_file)
        if closed2 is not None:
            self.logger.warning(f"Unable to open {self.hdf5_file!r}, "
                                "file status was forcibly reset")

        i, j, key_list, acceptance = self._restart_from_hdf5()
        cls = type(self)
        key: Any = key_list
        if not cls.HAS_LOOP:
            key = key_list[0]
            acceptance = acceptance[..., 0]

        # A previous run may have stopped early; continue only if the stop criteria allow it
        self._set_hdf5_attrs(stop_reason="")
        self.stop_reason = None
        if self._stop_after_iteration(i, j):
            return

        # Finish the current set of sub-iterations
        j += 1
        for omega in range(j, self.sub_iter_len):
            key = self.do_inner(i, omega, acceptance, key)
            if self._stop_after_iteration(i, omega):
                return
        self._end_super_iter(acceptance)
        if self._stop_after_super_iteration(i):
            return
        i += 1

        # And continue
        self(start=i, key_new=key)

    def _restart_from_hdf5(self) -> Tuple[int, int, List[Key], np.ndarray]:
        """Read and process the .hdf5 file for :meth:`ARMC.restart`."""
        with h5py.File(self.hdf5_file, 'r', libver='latest') as f:
            i, j = f.attrs['super-iteration'], f.attrs['sub-iteration']
            ij = j + 1 + i * f['param'].shape[1]  # The number of finished iterations
            if i < 0:
                raise ValueError(f'i: {i.__class__.__name__} = {i}')
            self.logger.info('Restarting ARMC procedure from super-iteration '
                             f'{i} & sub-iteration {j}')

            # Set the (to-be incremented) auxiliary error for each parameter
            param = f['param'][:i+1]
            param = param.reshape((-1,) + param.shape[2:])
            aux_error = self._aux_error_from_hdf5(f['aux_error'][:i+1])
            for keys, err in zip(param[:ij], aux_error[:ij]):
                if self.HAS_LOOP:
                    for _k, _err in zip(keys, err):
                        self[tuple(_k)] = _err
                else:
                    # All parameter columns hold the same parameters (one per molecule)
                    self[tuple(keys[0])] = err

            # Apply aforementioned increments
            aux_error_mod = f['aux_error_mod'][:i+1]
            aux_error_mod = aux_error_mod.reshape((-1,) + aux_error_mod.shape[2:])
            for _arr in aux_error_mod[:ij]:
                for arr in (_arr if self.HAS_LOOP else _arr[:1]):
                    *_k, err = arr
                    try:
                        self[tuple(_k)] += err
                    except KeyError:
                        pass

            self._restore_best_error(f, aux_error[:ij])

            # Restore the settings that ARMC-PT swaps between parameter sets.
            # Files written before Auto-FOX stored these keep the settings of the input.
            self.phi.phi = f['phi'][i]
            if 'a_target' in f.attrs:
                self.phi.a_target[:] = f.attrs['a_target']
                self.phi.gamma[:] = f.attrs['gamma']
                self.param.move_range[:] = f.attrs['move_range']
            acceptance = f['acceptance'][:i+1]

            # Continue with the random moves where the previous run stopped.
            # Files written before Auto-FOX stored this state get freshly seeded moves.
            if 'rng_state' in f.attrs:
                self.rng.bit_generator.state = json.loads(f.attrs['rng_state'])

            # Find the last accepted parameters and continue from those.
            # `f['param'][i, j]` holds the latest *proposed* parameters, which may be rejected.
            final_key: List[Key] = self._find_restart_key(acceptance, param[:ij])
            self.param.param[:] = np.array(final_key, dtype=np.float64).T
            self.param.param_old[:] = self.param.param
        return i, j, final_key, acceptance[i]

    def _restore_best_error(self, f: h5py.File, aux_error: np.ndarray) -> None:
        """Restore the lowest error found so far, for the stop criteria.

        **aux_error** holds the errors of all finished iterations.

        """
        totals = np.array([np.min(self._error_totals(err)) for err in aux_error])
        self._best_error = float(f.attrs.get('initial_error', np.inf))
        self._best_kappa = -1
        if len(totals) and np.min(totals) < self._best_error:
            self._best_error = float(np.min(totals))
            self._best_kappa = int(np.argmin(totals)) // f['param'].shape[1]

    def _aux_error_from_hdf5(self, aux_error: np.ndarray) -> np.ndarray:
        """Convert the auxiliary errors of the .hdf5 file into the layout of :meth:`get_aux_error`.

        The .hdf5 file stores them in the order of the PES descriptors, then the molecules
        (:class:`ARMC`) or parameter sets (:class:`ARMCPT`).

        """
        n = len(self.phi) if self.HAS_LOOP else len(self.molecule)
        aux_error = aux_error.reshape(-1, aux_error[0, 0].size // n, n)
        return np.swapaxes(aux_error, -2, -1)

    @staticmethod
    def _find_restart_key(acceptance: np.ndarray, param: np.ndarray) -> List[Key]:
        """Construct a key for the last parameter that was accepted."""
        accept = acceptance.reshape(-1, *acceptance.shape[2:])[:len(param)]
        if not accept.any():
            raise RuntimeError('Not a single successful MD-calculation was found; '
                               'restarting is not possible')

        j_array = -accept[::-1].argmax(axis=0)
        j_array -= 1
        return [tuple(param[i, j]) for j, i in enumerate(j_array)]
