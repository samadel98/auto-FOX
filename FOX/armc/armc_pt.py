"""A module for performing Addaptive Rate Monte Carlo Parallel Tempering (ARMC-PT) forcefield parameter optimizations.

Index
-----
.. currentmodule:: FOX.armc
.. autosummary::
    ARMCPT

API
---
.. autoclass:: ARMCPT
    :members:

"""  # noqa: E501

from typing import (
    Tuple, Dict, Mapping, Iterable, List, Sequence, overload, Any, TYPE_CHECKING, Callable,
    ClassVar, Optional,
    Literal,
)

import numpy as np

from .armc import ARMC
from ..type_hints import ArrayOrScalar

if TYPE_CHECKING:
    from ..classes import MultiMolecule
    from ..io import PSFContainer
else:
    from ..type_alias import MultiMolecule, PSFContainer

__all__ = ['ARMCPT']

PesDict = Dict[str, ArrayOrScalar]
PesMapping = Mapping[str, ArrayOrScalar]

MolList = List[MultiMolecule]
MolIter = Iterable[MultiMolecule]

Key = Tuple[float, ...]
SwapFunc = Callable[[np.ndarray, "ARMCPT"], Iterable[Tuple[int, int]]]


def swap_random(acceptance: np.ndarray, armcpt: "ARMCPT",
                weighted: bool = False) -> List[Tuple[int, int]]:
    r"""Swap the :math:`\phi` and move range of two between forcefield parameter sets.

    The two main parameters are the acceptance rate
    :math:`\boldsymbol{\alpha} \in \mathbb{R}^{(n,)}` and target acceptance rate
    :math:`\boldsymbol{\alpha^{t}} \in \mathbb{R}^{(n,)}`,
    both vector's elements being the :math:`[0,1]` interval.

    The (weighted) probability distribution :math:`\boldsymbol{p} \in \mathbb{R}^{(n,)}` is
    consequently used for identifying which two forcefield parameter sets
    are swapped.

    .. math::

        \hat{\boldsymbol{p}} = |\boldsymbol{\alpha} - \boldsymbol{\alpha^{t}}|^{-1}

        \boldsymbol{p} = \frac{\hat{\boldsymbol{p}}}{\sum^n_{i} {\hat{\boldsymbol{p}}}_{i}}

    Parameters
    ----------
    acceptance : :class:`np.ndarray[np.bool_] <numpy.ndarray>`, shape :math:`(n, m)`
        A 2D boolean array with acceptance rates over
        the course of the last super-iteration.
    armc : :class:`FOX.armc.ARMCPT`
        An ARMCPT instance.
    weighted : :class:`bool`
        If ``True``, identify the to-be swapped parameters via a
        weighted probability distribution.

    Returns
    -------
    :data:`List[Tuple[int, int]]<typing.List>`
        A list of 2-tuples with to-be swapped indices.

    """
    self = armcpt
    if weighted:
        _p = np.abs(acceptance.mean(axis=0) - self.phi.a_target)
        if 0 in _p:
            p = (_p == 0).astype(np.float64)
        else:
            p = _p**-1
    else:
        p = np.ones(acceptance.shape[1])
    p /= p.sum()  # normalize

    idx_range = np.arange(len(p))
    idx1 = self.rng.choice(idx_range, p=p)
    idx2 = self.rng.choice(idx_range, p=p)

    if idx1 != idx2:
        self.logger.info(f"Swapping parameter sets {idx1} and {idx2}")
        return [(int(idx1), int(idx2))]
    else:
        self.logger.info("Preserving all parameter sets")
        return []


class ARMCPT(ARMC):
    """An :class:`ARMC` subclass implementing a parallel tempering procedure."""

    HAS_LOOP: ClassVar[Literal[True]] = True  # type: ignore[assignment]

    def __init__(self, swapper: SwapFunc = swap_random, **kwargs: Any) -> None:
        r"""Initialize an :class:`ARMCPT` instance.

        Parameters
        ----------
        swapper : :data:`Callable[[ndarray, ARMCPT], Iterable[Tuple[int, int]]<collections.abc.Callable>`
            A callable for identifying the to-be swapped parameter.
            Should return an iterable with 2-tuples of to-be swapped indices.
        \**kwargs : :data:`~typing.Any`
            Further keyword arguments for the :class:`ARMC` and
            :class:`MonteCarloABC` superclasses.

        """  # noqa: E501
        super().__init__(**kwargs)
        self.param._is_independent = True
        self.swap_phi: SwapFunc = swapper
        if len(self.phi.phi) <= 1:
            raise ValueError("{self.__class__.__name__!r} requires 'phi.phi' to "
                             "contain more than 1 element")
        elif len(self._molecule) > 1:
            raise NotImplementedError

    def acceptance(self) -> np.ndarray:
        """Create an empty 2D boolean array for holding the acceptance."""
        shape = (self.sub_iter_len, len(self.phi.phi))
        return np.zeros(shape, dtype=bool)

    # ARMCPT works on a list of keys (one per parameter set), where ARMC works on a single key
    @overload  # type: ignore[override]
    def _parse_call(self, start: None = ..., key_new: None = ...) -> List[Key]: ...
    @overload
    def _parse_call(self, start: int, key_new: Iterable[Key]) -> List[Key]: ...
    def _parse_call(self, start=None, key_new=None):  # noqa: E301
        """Parse the arguments of :meth:`__call__` and prepare the first key."""
        ret = super()._parse_call(start, key_new)
        if start is key_new is None:
            self[ret] = self[ret][0]
            i = len(self.param.param.columns)
            return i * [ret]
        else:
            return list(ret)

    @overload  # type: ignore[override]
    def __call__(self, start: None = ..., key_new: None = ...) -> None: ...
    @overload
    def __call__(self, start: int, key_new: Key) -> None: ...
    def __call__(self, start=None, key_new=None):  # noqa: E301
        """Initialize the Addaptive Rate Monte Carlo procedure."""
        key_new = self._parse_call(start, key_new)
        start_ = start if start is not None else 0
        self._main_loop(start_, key_new)

    def _error_totals(self, aux_error: np.ndarray) -> np.ndarray:
        """Return the total error of each parameter set in **aux_error**."""
        return np.atleast_1d(np.sum(aux_error, axis=-1))

    def _end_super_iter(self, acceptance: np.ndarray) -> None:
        r"""Update :math:`\phi` and swap parameter sets at the end of a super-iteration."""
        self.phi.update(acceptance)
        idx_iter = self.swap_phi(acceptance, self)
        self._swap_phi(idx_iter)

    def do_inner(  # type: ignore[override]
        self, kappa: int, omega: int, acceptance: np.ndarray, key_old: Sequence[Key],
    ) -> List[Key]:
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
        _key_new = [self._do_inner1(key, idx=i) for i, key in enumerate(key_old)]

        # Step 2: Calculate PES descriptors
        pes_new, pes_validation, mol_list = self._do_inner2()

        # Step 3: Evaluate the auxiliary error; accept if the new parameter set lowers the error
        error_change, aux_new, aux_validation = self._do_inner3(pes_new, pes_validation, key_old)
        accept: np.ndarray = error_change < 0
        self._update_best_error(aux_new, kappa)

        # Step 4: Update the auxiliary error history, apply phi & update job settings
        acceptance[omega] = accept
        key_new = self._do_inner4(accept, error_change, aux_new,
                                  _key_new, key_old, kappa, omega)

        # Step 5: Export the results to HDF5
        self._do_inner5(mol_list, accept, aux_new, aux_validation,
                        pes_new, pes_validation, kappa, omega)
        return key_new

    def _do_inner3(  # type: ignore[override]
        self, pes_new: PesMapping, pes_validation: PesMapping, key_old: Iterable[Key],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the auxiliary error; accept if the new parameter set lowers the error."""
        aux_new = self.get_aux_error(pes_new)
        aux_validation = self.get_aux_error(pes_validation, validation=True)
        aux_old = np.array([self[k] for k in key_old])
        error_change = (aux_new - aux_old).sum(axis=-1)
        return error_change, aux_new, aux_validation

    def _do_inner4(  # type: ignore[override]
        self, accept: Iterable[bool], error_change: Iterable[float],
        aux_new: Iterable[np.ndarray],
        key_new: Iterable[Key], key_old: Iterable[Key],
        kappa: int, omega: int,
    ) -> List[Key]:
        """Update the auxiliary error history, apply phi & update job settings."""
        ret = []
        enumerator = enumerate(zip(key_new, key_old, accept, error_change, aux_new))
        for i, (k_new, k_old, acc, err_change, _aux_new) in enumerator:
            err_round = round(err_change, 4)
            aux_round = round(_aux_new.sum(), 4)
            epilog = f'error_change = {err_round}; error = {aux_round}'

            if acc:
                self.logger.info(f"Accepting move {(kappa, omega)}: {epilog}")
                self[k_new] = self.apply_phi(_aux_new, idx=i)
                self.param.param_old[i] = self.param.param[i]
                ret.append(k_new)
            else:
                self.logger.info(f"Rejecting move {(kappa, omega)}: {epilog}")
                self[k_new] = _aux_new
                self[k_old] = self.apply_phi(self[k_old], idx=i)
                ret.append(k_old)

        return ret

    def _swap_phi(self, idx_iter: Iterable[Tuple[int, int]]) -> None:
        """Swap the array-elements **idx1** and **idx2** of four :class:`ARMCPT` attributes.

        Affects the following attributes:

        * :attr:`ARMCPT.phi.phi<FOX.armc.PhiUpdater.phi>`
        * :attr:`ARMCPT.phi.a_target<FOX.armc.PhiUpdater.a_target>`
        * :attr:`ARMCPT.phi.gamma<FOX.armc.PhiUpdater.gamma>`
        * :attr:`ARMCPT.param.move_range<FOX.armc.ParamMapping.move_range>`

        """
        for idx1, idx2 in idx_iter:
            i = [idx1, idx2]
            j = [idx2, idx1]

            self.phi.phi[i] = self.phi.phi[j]
            self.phi.a_target[i] = self.phi.a_target[j]
            self.phi.gamma[i] = self.phi.gamma[j]
            self.param.move_range[i] = self.param.move_range[j]

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
        ret = super().to_yaml_dict(path=path, folder=folder, logfile=logfile, psf=psf)

        # Convert lists of settings into just settings
        for k, v in ret['job'].items():
            if k in {'molecule', 'type', 'lattice'}:
                continue
            v['settings'] = v['settings'][0]

        for k in ('pes', 'pes_validation'):
            for _k, v in ret[k].items():
                ret[k][_k]["weight"] = ret[k][_k]["weight"][:1]
        return ret
