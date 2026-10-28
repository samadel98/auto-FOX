"""A module for optimizing forcefield parameters with the CMA-ES.

Index
-----
.. currentmodule:: FOX.armc
.. autosummary::
    CMAES

API
---
.. autoclass:: CMAES
    :members:

"""

from __future__ import annotations

import copy
import math
import pickle
import warnings
from typing import (
    Any, ClassVar, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple,
    TYPE_CHECKING, cast, overload,
)

import h5py
import numpy as np

from .armc import ARMC
from ..io.hdf5_utils import create_hdf5, hdf5_availability
from ..type_hints import ArrayOrScalar

if TYPE_CHECKING:
    import cma
    from ..io.read_psf import PSFContainer
else:
    from ..type_alias import PSFContainer

__all__ = ['CMAES']

Key = Tuple[float, ...]
PesMapping = Mapping[str, ArrayOrScalar]


def _import_cma() -> Any:
    """Import :mod:`cma`, without its warning about a missing matplotlib."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Could not import matplotlib',
                                category=UserWarning)
        import cma
    return cma


class _Randn:
    """A picklable ``randn`` function for :mod:`cma` that draws from a :class:`~numpy.random.Generator`.

    It is pickled together with the CMA-ES state,
    so a restarted optimization draws the same candidates as an uninterrupted one.

    """  # noqa: E501

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def __call__(self, *shape: int) -> np.ndarray:
        return self.rng.standard_normal(shape)


class CMAES(ARMC):
    r"""Optimize forcefield parameters with the covariance matrix adaptation evolution strategy.

    Every iteration is a generation of the CMA-ES (Hansen, 2016, `arXiv:1604.00772`_):
    :attr:`popsize` candidate parameter sets are drawn from a multivariate normal distribution
    and their MD jobs are run (side by side with ``n_jobs > 1``),
    after which the mean, step size and covariance matrix of the distribution are updated
    from the best candidates.
    The CMA-ES itself is provided by the :mod:`cma` package.

    The parameters are optimized relative to their initial values:
    a parameter :math:`p` with initial value :math:`p_0 \ne 0` is represented by
    :math:`p / |p_0|`, so :attr:`sigma` is the initial step size as a fraction of
    the initial values.
    The net charge and the charge constraints (*e.g.* ``Cd == -2 * $LIGAND``) are linear
    equations in the charges, so the CMA-ES only varies the charges along the directions
    that keep them: every candidate has exactly the net charge and constraints of the input,
    without rebalancing the charges afterwards.
    The bounds of the parameters (*e.g.* ``0 < Cd < 2``) are respected;
    candidates with a charge outside its bounds are drawn anew.

    The first candidate of the first generation holds the initial parameters,
    so their error is known (and stored in the ``"initial_error"`` attribute of the .hdf5 file)
    without any extra MD job.

    The .hdf5 file has the layout of :class:`ARMCPT`, with one parameter set per candidate.
    For :class:`CMAES` the ``"acceptance"`` dataset marks the candidates that were
    selected for updating the distribution and the ``"phi"`` dataset holds the step size
    of the distribution. The ``"cmaes_state"`` dataset holds the (pickled) state of
    the CMA-ES, from which a calculation is restarted, and the ``"cmaes_offset"`` and
    ``"cmaes_transform"`` datasets hold the map from the CMA-ES to the parameters.

    Only a single molecule is supported; :class:`CMAES` does not do state-averaged fits.

    .. _`arXiv:1604.00772`: https://arxiv.org/abs/1604.00772

    Parameters
    ----------
    popsize : :class:`int`, optional
        The number of candidates per generation, :math:`\lambda`.
        Defaults to :math:`4 + \lfloor 3 \ln n \rfloor`, :math:`n` being the number of
        variable parameters minus the number of charge equations
        (the net charge and the charge constraints).
        A multiple of ``n_jobs`` uses the parallel jobs best.
    sigma : :class:`float`
        The initial step size, as a fraction of the initial parameter values.
    \**kwargs : :data:`~typing.Any`
        Further keyword arguments for the :class:`ARMC` and :class:`MonteCarloABC` superclasses.

    """

    HAS_LOOP: ClassVar[Literal[True]] = True  # type: ignore[assignment]

    #: The maximum number of times a candidate with a charge outside its bounds is redrawn.
    MAX_REDRAW: ClassVar[int] = 1000

    _PRIVATE_ATTR = frozenset({'_rng', '_start_time', '_es'})  # type: ignore

    popsize: Optional[int]
    sigma: float

    def __init__(self, popsize: Optional[int] = None, sigma: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if len(self.molecule) > 1:
            raise NotImplementedError(f"{type(self).__name__!r} does not support more than "
                                      "one molecule (state-averaged fits)")
        elif len(self.phi) > 1:
            raise ValueError(f"{type(self).__name__!r} expected a single 'phi.phi' value; "
                             f"observed number of values: {len(self.phi)}")

        if popsize is not None:
            if isinstance(popsize, (bool, np.bool_)):
                raise TypeError("'popsize' expected an integer; observed type: 'bool'")
            popsize = int(popsize)
            if popsize < 2:
                raise ValueError(f"'popsize' expected an integer larger than 1; "
                                 f"observed value: {popsize!r}")
        self.popsize = popsize

        sigma = float(sigma)
        if not (0 < sigma < np.inf):
            raise ValueError(f"'sigma' expected a finite positive float; observed value: {sigma!r}")
        self.sigma = sigma

        self._phi_yaml = self.phi.to_yaml_dict()
        self._n_candidates: Optional[int] = None
        self._es: Optional[cma.CMAEvolutionStrategy] = None

        # The variable parameters of a candidate `x` of the CMA-ES are `offset + transform @ x`
        self._offset: np.ndarray = np.zeros(0)
        self._transform: np.ndarray = np.zeros((0, 0))

    @property
    def n_candidates(self) -> int:
        r"""The number of candidates per generation, :math:`\lambda`."""
        if self._n_candidates is not None:
            return self._n_candidates
        elif self.popsize is not None:
            return self.popsize
        n = len(self._get_transform()[2])  # The number of coordinates of the CMA-ES
        return 4 + int(3 * math.log(max(n, 1)))

    def _variable_index(self) -> Any:
        """Return the index of the variable (*i.e.* non-frozen) parameters."""
        frozen = self.param.metadata[0, 'frozen'].astype(bool)
        return self.param.param.index[~frozen.values]

    # Turn the single parameter set of the input into one parameter set per candidate

    def _expand(self) -> None:
        """Create one parameter set, job and PES descriptor per candidate."""
        if self._n_candidates is not None:
            return

        n_var = len(self._variable_index())
        if n_var == 0:
            raise ValueError(f"{type(self).__name__!r} requires at least one variable parameter")
        n = self.n_candidates

        param = self.param
        param.set_n_columns(n)
        param.move_range = np.repeat(param.move_range[:1], n, axis=0)
        param._is_independent = True
        if param._net_charge is not None:
            net_charge = np.repeat(param._net_charge[:1], n)
            net_charge.setflags(write=False)
            param._net_charge = net_charge

        # `phi` is not used by the CMA-ES; it holds the step size of the distribution instead
        self.phi.phi = np.full(n, self.sigma)
        self.phi._gamma = np.repeat(self.phi.gamma[:1], n)
        self.phi._a_target = np.repeat(self.phi.a_target[:1], n)

        data: Dict[str, List[Dict[str, Any]]] = {}
        for name, jobs in self.package_manager.items():
            job = jobs[0]
            data[name] = [
                {**job, 'settings': copy.deepcopy(job['settings']),
                 'molecule': job['molecule'].copy()}
                for _ in range(n)
            ]
        self.package_manager.data = data  # type: ignore[assignment]

        for attr in ('pes', 'pes_validation'):
            pes: Dict[str, Any] = {}
            for key, func in getattr(self, attr).items():
                name, _, _ = key.rpartition('.')
                pes.update((f'{name}.{i}', func) for i in range(n))
            setattr(self, attr, pes)
        self._n_candidates = n

    def _get_transform(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Construct the map between the CMA-ES and the variable parameters.

        A candidate :math:`x` of the CMA-ES holds the parameters :math:`p = o + T x`.
        The parameters that are not charges get :math:`p_i = |p_{0,i}| x_i`.
        For the charges, :math:`o` is the initial charges (moved as little as possible
        to meet the net charge and the charge constraints, :math:`A q = b`, exactly)
        and the columns of :math:`T` span the null space of :math:`A`,
        so every candidate keeps the net charge and the constraints.

        Returns
        -------
        :class:`tuple[np.ndarray, np.ndarray, np.ndarray] <tuple>`
            The offset :math:`o`, the transformation matrix :math:`T` and
            the initial candidate :math:`x_0`.

        """
        param = self.param
        idx = self._variable_index()
        p0 = param.param.loc[idx, 0].values.astype(np.float64)
        scale = np.where(p0 != 0, np.abs(p0), 1.0)

        net_charge = param._net_charge
        if net_charge is None:
            charge_keys: List[Tuple[str, str]] = []
        else:
            charge_keys = list(dict.fromkeys(
                k[:2] for k in idx if k[1] in param.CHARGE_LIKE  # type: ignore[attr-defined]
            ))
        is_charge = np.array([k[:2] in charge_keys for k in idx], dtype=np.bool_)

        # Parameters other than charges: a diagonal block
        n_var = len(idx)
        offset = np.zeros(n_var)
        columns = [np.eye(n_var)[:, i] * scale[i] for i in np.flatnonzero(~is_charge)]
        x0 = list(p0[~is_charge] / scale[~is_charge])

        # Charges: the null space of their linear constraints
        for key in charge_keys:
            charge = param.param.loc[key, 0].astype(np.float64)
            count = param.metadata.loc[key, (0, 'count')].astype(np.float64)
            frozen = param.metadata.loc[key, (0, 'frozen')].values.astype(bool)

            # The net charge and, for every block of a constraint after the first,
            # `block_n @ q - block_1 @ q == 0`
            rows = [count.values]
            rhs = [float(cast(np.ndarray, net_charge)[0])]
            blocks = [
                s.reindex(charge.index, fill_value=0).values.astype(np.float64)
                for s in (param.constraints.get(key) or ())
            ]
            rows += [b - blocks[0] for b in blocks[1:]]
            rhs += [0.0] * len(blocks[1:])
            a = np.array(rows)
            b = np.array(rhs) - a[:, frozen] @ charge.values[frozen]
            a = a[:, ~frozen]

            pos = [idx.get_loc((*key, atom)) for atom in charge.index[~frozen]]
            a_scaled = a * scale[pos]
            y0 = charge.values[~frozen] / scale[pos]
            y = y0 + np.linalg.lstsq(a_scaled, b - a_scaled @ y0, rcond=None)[0]
            if not np.allclose(a_scaled @ y, b, rtol=0, atol=1e-6):
                raise ValueError(f"The {key[1]!r} parameters cannot meet the net charge "
                                 f"and the charge constraints at the same time")
            offset[pos] = y * scale[pos]

            _, sv, vt = np.linalg.svd(a_scaled)
            tol = sv.max(initial=0) * max(a_scaled.shape) * np.finfo(np.float64).eps
            for v in vt[np.count_nonzero(sv > tol):]:
                col = np.zeros(n_var)
                col[pos] = v * scale[pos]
                columns.append(col)
                x0.append(0.0)

        transform = np.array(columns, dtype=np.float64).reshape(-1, n_var).T
        return offset, transform, np.array(x0, dtype=np.float64)

    def _to_param(self, x: np.ndarray) -> np.ndarray:
        """Return the variable parameters of the CMA-ES candidate **x**: ``offset + transform @ x``.

        The product is summed column by column rather than with BLAS, whose summation order
        can depend on the memory layout of **transform** (which differs after reading it from
        the .hdf5 file), so a restarted run reproduces the parameters bit for bit.

        """  # noqa: E501
        ret = self._offset.copy()
        for column, x_i in zip(self._transform.T, x):
            ret += column * x_i
        return ret

    def _bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the lower and upper bounds of the variable parameters."""
        idx = self._variable_index()
        lower = self.param.metadata.loc[idx, (0, 'min')].values.astype(np.float64)
        upper = self.param.metadata.loc[idx, (0, 'max')].values.astype(np.float64)
        return lower, upper

    def _new_es(self) -> cma.CMAEvolutionStrategy:
        """Construct the CMA-ES, starting from the initial parameters."""
        self._offset, self._transform, x0 = self._get_transform()
        if x0.size == 0:
            raise ValueError(f"{type(self).__name__!r} requires at least one parameter that "
                             "is free to vary; the net charge and charge constraints fix all "
                             "variable charges")
        lower, upper = self._bounds()
        if not self._in_bounds(self._to_param(x0), lower, upper):
            raise ValueError("The initial parameters are not within their bounds "
                             "after meeting the net charge and charge constraints")

        # The bounds of the parameters that are not charges are handled by the CMA-ES;
        # the coordinates of a diagonal column of `transform` map onto a single parameter
        options: Dict[str, Any] = {
            'popsize': self.n_candidates,
            'randn': _Randn(self.rng),
            'verbose': -9,
            'verb_log': 0,
            'verb_disp': 0,
        }
        x_lower = np.full_like(x0, -np.inf)
        x_upper = np.full_like(x0, np.inf)
        for j, col in enumerate(self._transform.T):
            (nonzero,) = np.nonzero(col)
            if len(nonzero) == 1 and self._offset[nonzero[0]] == 0:
                i = nonzero[0]
                x_lower[j], x_upper[j] = sorted([lower[i] / col[i], upper[i] / col[i]])
        if np.isfinite(x_lower).any() or np.isfinite(x_upper).any():
            options['bounds'] = [x_lower.tolist(), x_upper.tolist()]

        es = _import_cma().CMAEvolutionStrategy(x0, self.sigma, options)
        es.inject([x0], force=True)  # The first candidate holds the initial parameters
        return es

    @staticmethod
    def _in_bounds(p: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> bool:
        """Check if the parameters **p** are within their bounds, up to round-off errors."""
        with np.errstate(invalid='ignore'):
            tol_lower = lower - 1e-9 * np.maximum(np.abs(lower), 1)
            tol_upper = upper + 1e-9 * np.maximum(np.abs(upper), 1)
        return bool(np.all(p >= tol_lower) and np.all(p <= tol_upper))

    # Running the optimization

    @overload  # type: ignore[override]
    def __call__(self, *, start: None = ..., key_new: None = ...) -> None: ...
    @overload
    def __call__(self, *, start: int, key_new: Iterable[Key]) -> None: ...
    def __call__(self, *, start=None, key_new=None):  # noqa: E301
        """Run the CMA-ES optimization."""
        self._expand()
        key_new = self._parse_call(start, key_new)
        start_ = start if start is not None else 0
        self._main_loop(start_, key_new)

    @overload  # type: ignore[override]
    def _parse_call(self, start: None = ..., key_new: None = ...) -> List[Key]: ...
    @overload
    def _parse_call(self, start: int, key_new: Iterable[Key]) -> List[Key]: ...
    def _parse_call(self, start=None, key_new=None):  # noqa: E301
        """Parse the arguments of :meth:`__call__` and prepare the CMA-ES."""
        if start is None:
            create_hdf5(self.hdf5_file, self)
            self._es = self._new_es()
            with h5py.File(self.hdf5_file, 'r+', libver='latest') as f:
                f.create_dataset('cmaes_offset', data=self._offset)
                f.create_dataset('cmaes_transform', data=self._transform)
                f.create_dataset('cmaes_state', shape=(0,), maxshape=(None,),
                                 dtype=np.uint8, chunks=True)
            return self._keys()
        elif key_new is None:
            raise TypeError("'key_new' cannot be None if 'start' is None")
        return list(key_new)

    def _keys(self) -> List[Key]:
        """Return the keys of all candidates."""
        return [tuple(v) for _, v in self.param.param.items()]

    def acceptance(self) -> np.ndarray:
        """Create an empty 2D boolean array for holding the selected candidates."""
        return np.zeros((self.sub_iter_len, self.n_candidates), dtype=bool)

    def _error_totals(self, aux_error: np.ndarray) -> np.ndarray:
        """Return the total error of each candidate in **aux_error**."""
        return np.atleast_1d(np.sum(aux_error, axis=-1))

    def _end_super_iter(self, acceptance: np.ndarray) -> None:
        """Nothing to do at the end of a super-iteration; the CMA-ES updates every iteration."""
        pass

    def do_inner(  # type: ignore[override]
        self, kappa: int, omega: int, acceptance: np.ndarray, key_old: Sequence[Key],
    ) -> List[Key]:
        r"""Run a single generation of the CMA-ES.

        Parameters
        ----------
        kappa : :class:`int`
            The super-iteration, :math:`\kappa`, in :meth:`CMAES.__call__`.
        omega : :class:`int`
            The sub-iteration, :math:`\omega`, in :meth:`CMAES.__call__`.
        acceptance : :class:`np.ndarray[np.bool_] <numpy.ndarray>`
            An array with the selected candidates over the course of the latest super-iteration.
        key_old : :class:`Sequence[tuple[float, ...]] <collections.abc.Sequence>`
            The candidates of the previous generation.

        Returns
        -------
        :class:`list[tuple[float, ...]] <list>`
            The candidates of this generation.

        """
        es = self._es
        assert es is not None
        first_generation = es.countiter == 0

        # Step 1: Draw the candidates
        x_list = self._ask()
        self.package_manager.update_settings(self.param.get_cp2k_dicts())
        key_new = self._keys()

        # Step 2: Calculate PES descriptors
        pes_new, pes_validation, mol_list = self.get_pes_descriptors()

        # Step 3: Evaluate the auxiliary error
        self.logger.info("Calculating auxiliary error")
        aux_new = self.get_aux_error(pes_new)
        aux_validation = self.get_aux_error(pes_validation, validation=True)
        error = self._error_totals(aux_new)
        for key, err in zip(key_new, aux_new):
            self[key] = err
        if first_generation:
            self._set_hdf5_attrs(initial_error=error[0])
        self._update_best_error(aux_new, kappa)

        # Step 4: Update the distribution with the best candidates
        accept = np.zeros(len(error), dtype=np.bool_)
        if np.isfinite(error).any():
            es.tell(x_list, error.tolist())
            accept[np.argsort(error, kind='stable')[:es.sp.weights.mu]] = True
            self.logger.info(f"Generation {(kappa, omega)}: lowest error = "
                             f"{round(np.min(error), 4)}; step size = {es.sigma:.4g}")
        else:
            self.logger.warning(f"Generation {(kappa, omega)}: all MD jobs crashed; "
                                "drawing new candidates")
        acceptance[omega] = accept
        self.phi.phi = np.full(len(error), es.sigma)
        self._set_mean()

        # Step 5: Export the results to HDF5
        self.to_hdf5(mol_list, accept, aux_new, aux_validation,
                     pes_new, pes_validation, kappa, omega)
        return key_new

    def _ask(self) -> List[np.ndarray]:
        """Draw the candidates of a generation and set them as parameters."""
        es = self._es
        assert es is not None
        ret = []
        for i, x in enumerate(es.ask()):
            for _ in range(self.MAX_REDRAW):
                if self._set_candidate(i, x):
                    ret.append(x)
                    break
                x = es.ask(1)[0]
            else:
                raise RuntimeError("Failed to draw a candidate with all charges within their "
                                   f"bounds in {self.MAX_REDRAW} attempts")
        return ret

    def _set_candidate(self, i: int, x: np.ndarray) -> bool:
        """Set candidate **x** as the parameters of column **i**.

        Returns :data:`False` if a parameter of **x** is outside its bounds.

        """
        p = self._to_param(x)
        lower, upper = self._bounds()
        if not self._in_bounds(p, lower, upper):
            self.logger.debug(f"Redrawing candidate {i}: a charge is outside its bounds")
            return False
        self.param.param.loc[self._variable_index(), i] = np.clip(p, lower, upper)
        return True

    def _set_mean(self) -> None:
        """Store the mean of the distribution in :attr:`ParamMapping.param_old`."""
        es = self._es
        assert es is not None
        idx = self._variable_index()
        mean = self._to_param(es.result.xfavorite)
        for i in self.param.param_old.columns:
            self.param.param_old.loc[idx, i] = mean

    def _stop_after_iteration(self, kappa: int, omega: int) -> bool:
        """Check if the optimization should stop after iteration (**kappa**, **omega**).

        Besides the criteria of :attr:`stop`, this checks the termination criteria of
        the CMA-ES itself, such as the distribution having converged.

        """
        if super()._stop_after_iteration(kappa, omega):
            return True
        elif (kappa, omega) == (self.super_iter_len - 1, self.sub_iter_len - 1):
            return False
        termination = self._es.stop() if self._es is not None else {}
        if not termination:
            return False
        return self._set_stop_reason(f"the CMA-ES met its termination criteria "
                                     f"({', '.join(termination)})")

    def to_hdf5(self, mol_list: Any, accept: Any, aux_new: np.ndarray,
                aux_validation: np.ndarray, pes_new: PesMapping, pes_validation: PesMapping,
                kappa: int, omega: int) -> None:
        """Export the results of a generation and the state of the CMA-ES to the .hdf5 file.

        The state is written first, so that a crash in between leaves the iteration
        counters of the .hdf5 file (written last) at the previous generation.

        """
        state = np.frombuffer(pickle.dumps(self._es), dtype=np.uint8)
        hdf5_availability(self.hdf5_file)
        with h5py.File(self.hdf5_file, 'r+', libver='latest') as f:
            dset = f['cmaes_state']
            dset.resize(state.shape)
            dset[:] = state
        super().to_hdf5(mol_list, accept, aux_new, aux_validation,
                        pes_new, pes_validation, kappa, omega)

    # Restarting

    def restart(self) -> None:  # type: ignore[override]
        """Restart a previously started CMA-ES optimization."""
        self._expand()
        super().restart()

    def _restart_from_hdf5(self) -> Tuple[int, int, List[Key], np.ndarray]:
        """Read and process the .hdf5 file for :meth:`CMAES.restart`."""
        with h5py.File(self.hdf5_file, 'r', libver='latest') as f:
            i, j = int(f.attrs['super-iteration']), int(f.attrs['sub-iteration'])
            if i < 0:
                raise ValueError(f'i: {i.__class__.__name__} = {i}')
            ij = j + 1 + i * f['param'].shape[1]  # The number of finished iterations
            self.logger.info('Restarting CMA-ES procedure from super-iteration '
                             f'{i} & sub-iteration {j}')

            # The errors of all candidates so far
            param = f['param'][:i+1]
            param = param.reshape((-1,) + param.shape[2:])[:ij]
            aux_error = self._aux_error_from_hdf5(f['aux_error'][:i+1])[:ij]
            for keys, err in zip(param, aux_error):
                for k, e in zip(keys, err):
                    self[tuple(k)] = e
            self._restore_best_error(f, aux_error)

            # The CMA-ES continues from its state after the last finished generation
            self._offset = f['cmaes_offset'][:]
            self._transform = f['cmaes_transform'][:]
            _import_cma()
            self._es = pickle.loads(f['cmaes_state'][:].tobytes())
            self.phi.phi = f['phi'][i]
            acceptance = f['acceptance'][:i+1]

        self.param.param[:] = param[-1].T
        self._set_mean()
        return i, j, self._keys(), acceptance[i]

    def to_yaml_dict(  # type: ignore[override]
        self, *,
        path: str = '.',
        folder: str = 'MM_MD_workdir',
        logfile: str = 'armc.log',
        psf: Optional[Iterable[PSFContainer]] = None,
    ) -> Dict[str, Any]:
        """Convert a :class:`CMAES` instance into a .yaml readable by :func:`dict_to_armc`.

        Returns
        -------
        :data:`dict[str, Any] <dict>`
            A dictionary.

        """
        ret = super().to_yaml_dict(path=path, folder=folder, logfile=logfile, psf=psf)
        ret['monte_carlo']['popsize'] = self.popsize
        ret['monte_carlo']['sigma'] = self.sigma
        ret['phi'] = copy.deepcopy(self._phi_yaml)
        if self._n_candidates is None:
            return ret

        # Keep only the first candidate, as in the input
        ret['param']['move_range'] = ret['param']['move_range'][:1]
        for k, v in ret['job'].items():
            if k not in {'molecule', 'type', 'lattice'}:
                v['settings'] = v['settings'][:1]
        for k in ('pes', 'pes_validation'):
            for v in ret[k].values():
                v['kwargs'] = v['kwargs'][:1]
                v['weight'] = v['weight'][:1]
                if v['ref'] is not None:
                    v['ref'] = v['ref'][:1]
        return ret
