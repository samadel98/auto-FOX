"""Callables for post-processing :class:`MultiMolecule` instances produced by :class:`ARMC`.

Index
-----
.. currentmodule:: FOX.armc
.. autosummary::
    AtomsFromPSF

API
---
.. autoclass:: AtomsFromPSF

"""

from __future__ import annotations

from itertools import repeat
from typing import Iterable, List, Optional, TYPE_CHECKING, Mapping, Tuple, Any, Dict

if TYPE_CHECKING:
    from .armc import ARMC
    from ..io import PSFContainer
    from ..classes import MultiMolecule
else:
    from ..type_alias import ARMC, PSFContainer, MultiMolecule

__all__ = ['AtomsFromPSF']

AtomMapping = Mapping[str, Tuple[str, Any]]
AtomDict = Dict[str, Tuple[str, List[int]]]


class AtomsFromPSF:
    r"""A callable for updating the atoms of a :class:`MultiMolecule` instances.

    Examples
    --------
    .. code:: python

        >>> from typing import Callable
        >>> from FOX import PSFContainer, MultiMolecule

        >>> psf_list = [PSFContainer(...), PSFContainer(...), PSFContainer(...)]
        >>> atoms_from_psf: Callable = AtomsFromPSF.from_psf(*psf_list)

        >>> mol_list = [MultiMolecule(...), MultiMolecule(...), MultiMolecule(...)]
        >>> atoms_from_psf(None, mol_list)

    Parameters
    ----------
    \*atom_dict : :class:`dict[str, list[int]] <dict>`
        One or more dictionaries with atomic symbols as keys and
        lists of matching atomic indices as values.

    """

    @classmethod
    def from_psf(cls, *psf: PSFContainer) -> AtomsFromPSF:
        """Construct a :class:`AtomsFromPsf` instance from one or more :class:`PSFContainer`."""
        return cls(*[p.to_atom_alias_dict() for p in psf])

    def __init__(self, *atom_dict: AtomMapping) -> None:
        """Initialize the :class:`AtomsFromPsf` instance."""
        self.atom_dict = atom_dict

    def __call__(self, mol_list: Optional[Iterable[Optional[MultiMolecule]]],
                 mc: Optional[ARMC] = None) -> None:
        """Update the :attr:`MultiMolecule.atoms` of **mol_list**."""
        if mol_list is None:
            return
        # A single .psf file holds for all molecules: the copies of one molecule that
        # :class:`~FOX.armc.ARMCPT` and :class:`~FOX.armc.CMAES` run, one per parameter set
        atom_dicts = self.atom_dict if len(self.atom_dict) != 1 else repeat(self.atom_dict[0])
        for atom_dict, mol in zip(atom_dicts, mol_list):
            if mol is not None:  # `None` if the job crashed
                mol.atoms_alias = atom_dict
