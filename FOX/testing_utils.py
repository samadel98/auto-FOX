"""A module with utility functions for testing Auto-FOX.

Index
-----
.. currentmodule:: FOX.test_utils
.. autosummary::
    validate_mapping
    load_results
    compare_hdf5

API
---
.. autofunction:: validate_mapping
.. autofunction:: load_results
.. autofunction:: compare_hdf5

"""

import re
import warnings
from types import MappingProxyType
from os import listdir
from os.path import isdir, abspath
from pathlib import Path
from collections import abc
from typing import (
    Mapping, Optional, TypeVar, Type, Dict, Callable, Any, Union, Tuple, List,
    overload, TYPE_CHECKING, Container,
    Literal,
)

import h5py
import numpy as np

from scm.plams import Molecule
from qmflows import Settings as QmSettings
from qmflows.packages import CP2KMM_Result

from ._nanoutils import PathType, recursive_keys
from .functions.cp2k_utils import get_xyz_path

if TYPE_CHECKING:
    from qmflows.packages import Result
else:
    from .type_alias import Result

__all__ = ['validate_mapping', 'load_results', 'compare_hdf5']

KT = TypeVar('KT')
VT = TypeVar('VT')
RT = TypeVar('RT', bound=Result)


class AssertionWarning(Warning):
    """A :exc:`Warning` subclass for assertion-related warnings."""


class _Key:
    pass


def _assert_isinstance(obj: object, cls: Union[type, Tuple[type, ...]]) -> None:
    """Raise an :exc:`AssertionError` if **obj** is not an instance of **cls**."""
    __tracebackhide__ = True
    if not isinstance(obj, cls):
        raise AssertionError(f"expected an instance of {cls!r}; "
                             f"observed type: {obj.__class__.__name__!r}")


def _assert_is(obj: object, value: object) -> None:
    """Raise an :exc:`AssertionError` if **obj** is not **value**."""
    __tracebackhide__ = True
    if obj is not value:
        raise AssertionError(f"expected {value!r}; observed value: {obj!r}")


def _val_class(mapping: Mapping) -> None:
    """Validate the :class:`type` of **mapping**."""
    _assert_isinstance(mapping, abc.Mapping)


def _val_items(mapping: Mapping[KT, VT],
               k_type: Union[Type[KT], Tuple[Type[KT], ...]],
               v_type: Union[Type[VT], Tuple[Type[VT], ...]]) -> None:
    """Validate the :meth:`~collections.abc.Mapping.items` method of **mapping**."""
    items = mapping.items()
    _assert_isinstance(items, abc.ItemsView)
    for k, v in mapping.items():
        _assert_isinstance(k, k_type)
        _assert_isinstance(v, v_type)


def _val_keys(mapping: Mapping[KT, Any], k_type: Union[Type[KT], Tuple[Type[KT], ...]]) -> None:
    """Validate the :meth:`~collections.abc.Mapping.keys` method of **mapping**."""
    keys = mapping.keys()
    _assert_isinstance(keys, abc.KeysView)
    for k in keys:
        _assert_isinstance(k, k_type)


def _val_values(mapping: Mapping[Any, VT], v_type: Union[Type[VT], Tuple[Type[VT], ...]]) -> None:
    """Validate the :meth:`~collections.abc.Mapping.values` method of **mapping**."""
    values = mapping.values()
    _assert_isinstance(values, abc.ValuesView)
    for v in values:
        _assert_isinstance(v, v_type)


def _val_iter(mapping: Mapping[KT, Any], k_type: Union[Type[KT], Tuple[Type[KT], ...]]) -> None:
    """Validate the :meth:`~collections.abc.Mapping.__iter__` method of **mapping**."""
    iterator = iter(mapping)
    _assert_isinstance(iterator, abc.Iterator)
    for k in iterator:
        _assert_isinstance(k, k_type)


def _val_len(mapping: Mapping) -> None:
    """Validate the :meth:`~collections.abc.Mapping.__len__` method of **mapping**."""
    length = len(mapping)
    _assert_isinstance(length, int)


def _val_contains(mapping: Mapping[KT, Any], key: KT) -> None:
    """Validate the :meth:`~collections.abc.Mapping.__contains__` method of **mapping**."""
    contains_true = key in mapping
    contains_false = _Key in mapping
    _assert_is(contains_true, True)
    _assert_is(contains_false, False)


def _val_getitem(mapping: Mapping[KT, VT], key: KT,
                 v_type: Union[Type[VT], Tuple[Type[VT], ...]]) -> None:
    """Validate the :meth:`~collections.abc.Mapping.__getitem__` method of **mapping**."""
    v = mapping[key]
    _assert_isinstance(v, v_type)


def _val_get(mapping: Mapping[KT, VT], key: KT,
             v_type: Union[Type[VT], Tuple[Type[VT], ...]]) -> None:
    """Validate the :meth:`~collections.abc.Mapping.get` method of **mapping**."""
    v = mapping.get(key, default=True)  # type: ignore
    _v = mapping.get(_Key, default=True)  # type: ignore
    _assert_isinstance(v, v_type)
    _assert_is(_v, True)


Name = Literal['__class__', 'items', 'keys', 'values', '__iter__',
               '__len__', 'get', '__getitem__', '__contains__']


VALIDATION_MAP: Mapping[Name, Callable[..., None]] = MappingProxyType({
    '__class__': _val_class,
    'items': _val_items,
    'keys': _val_keys,
    'values': _val_values,
    '__iter__': _val_iter,
    '__len__': _val_len,
    'get': _val_get,
    '__getitem__': _val_getitem,
    '__contains__': _val_contains
})


def validate(name: Name, mapping: Mapping, *args: Any, **kwargs: Any) -> Optional[Exception]:
    """Perform the **name**-specific validation of **mapping**."""
    try:
        func = VALIDATION_MAP[name]
        func(mapping, *args, **kwargs)
    except Exception as ex:
        return ex
    else:
        return None


def validate_mapping(mapping: Mapping[KT, VT],
                     key_type: Optional[Union[Type[KT], Tuple[Type[KT], ...]]] = None,
                     value_type: Optional[Union[Type[VT], Tuple[Type[VT], ...]]] = None) -> None:
    """Validate an implementation of a :class:`~collections.abc.Mapping` protocol in **mapping**.

    Parameters
    ----------
    mapping : :class:`~collections.abc.Mapping`
        The to-be validated Mapping.

    key_type : :class:`type`, optional
        The type of the **mapping** keys.
        If not ``None`` it will be used for instance checking.

    value_type : :class:`type`, optional
        The type of the **mapping** values.
        If not ``None`` it will be used for instance checking.

    Exceptions
    ----------
    :exc:`AssertionError`
        Raised if the validation is unsuccessful.

    """
    k_type = object if key_type is None else key_type
    v_type = object if value_type is None else value_type

    cls = mapping.__class__.__name__

    err_dict: Dict[str, Optional[Exception]] = {
        f'{cls}.__class__': validate('__class__', mapping),
        f'{cls}.items': validate('items', mapping, k_type, v_type),
        f'{cls}.keys': validate('keys', mapping, k_type),
        f'{cls}.values': validate('values', mapping, v_type),
        f'{cls}.__iter__': validate('__iter__', mapping, k_type),
        f'{cls}.__len__': validate('__len__', mapping)
    }

    try:
        key = next(iter(mapping.keys()))
    except StopIteration as ex:  # In case **mapping** is empty
        err_dict[f'{cls}.get'] = None
        err_dict[f'{cls}.__getitem__'] = None
        err_dict[f'{cls}.__contains__'] = None
        warning = AssertionWarning("The passed mapping is empty; cannot validate "
                                   f"'{cls}.get', '{cls}.__getitem__' and '{cls}.__contains__'")
        warning.__cause__ = ex
        warnings.warn(warning)
    except Exception as ex:
        err_dict[f'{cls}.get'] = ex
        err_dict[f'{cls}.__getitem__'] = ex
        err_dict[f'{cls}.__contains__'] = ex
    else:
        err_dict[f'{cls}.get'] = validate('get', mapping, key, v_type)
        err_dict[f'{cls}.__getitem__'] = validate('__getitem__', mapping, key, v_type)
        err_dict[f'{cls}.__contains__'] = validate('__contains__', mapping, key)

    # Gather all failed tests
    exc_dict = {k: v for k, v in err_dict.items() if v is not None}
    if not exc_dict:
        return

    # Raise an AssertionError
    _, exc_old = next(iter(exc_dict.items()))
    raise AssertionError("Failed to validate the following 'mapping' attributes:\n"
                         f"{', '.join(repr(k) for k in exc_dict)}") from exc_old


@overload
def load_results(workdir: PathType, *, n: int = ...) -> List[Tuple[CP2KMM_Result, ...]]:
    ...
@overload  # noqa: E302
def load_results(workdir: PathType, *, result_type: Type[RT], n: int = ...) -> List[Tuple[RT, ...]]:
    ...
def load_results(workdir, *, result_type=CP2KMM_Result, n=1):  # noqa: E302
    """Construct a :class:`~qmflows.packages.packages.Result` instances for all jobs in the passed PLAMS working directory.

    Parameters
    ----------
    workdir : path-like
        A path-like object pointing to the PLAMS working directory.
    result_type : :class:`type` [:class:`~qmflows.packages.packages.Result`]
        The type of the to-be returned Result object.
    n : :class:`int`
        The length of each nested tuple.

    Returns
    -------
    :class:`List[Tuple[Result, ...]]]<typing.List>`
        A list of tuples with Result objects of the type specified in **result_type**.
        Each tuple is of length **n**.

    """  # noqa: E501
    if n <= 0:
        raise ValueError(f"'n' cannot be smaller than 1; observed value: {n!r}")
    workdir_path = Path(abspath(workdir))

    ret = []
    tmp = []
    i = -n
    file_pattern = re.compile(r"md(\.([0-9]+))?")
    for jobname in sorted(listdir(workdir_path)):
        if file_pattern.fullmatch(jobname) is None:
            continue

        plams_dir = workdir_path / jobname
        if not isdir(plams_dir):
            continue

        dill_path = plams_dir / f'{jobname}.dill'
        settings = QmSettings()  # Extracting the settings is too much work for now
        mol = Molecule(get_xyz_path(plams_dir))

        result = result_type(settings, mol, jobname, dill_path=dill_path,
                             plams_dir=plams_dir, work_dir=plams_dir, status='successful')

        tmp.append(result)
        i += 1
        if not i:
            ret.append(tuple(tmp))
            tmp = []
            i = -n

    if len(ret[-1]) != n:
        m = len(ret[-1])
        raise ValueError(f"len(ret[-1]) == {m}; {m} != {n!r}")
    return ret


def compare_hdf5(f1: h5py.Group, f2: h5py.Group, skip: Container[str] = frozenset({})) -> None:
    """Check if the datasets and attributes passed hdf5 groups are equivalent.

    The passed groups will be traversed recursively if necessary.

    Parameters
    ----------
    f1/f2 : :class:`h5py.Group`
        The to-be compared h5py groups.
    skip : :class:`Container[str]<collections.abc.Container>`
        A container with the names (:attr:`h5py.Dataset.name`) of to-be ignored datasets.

    Raises
    ------
    AssertionError
        Raised if a comparison evaluates to :data:`False`.

    """
    __tracebackhide__ = True
    keys1, keys2 = list(recursive_keys(f1)), list(recursive_keys(f2))
    if keys1 != keys2:
        raise AssertionError(f"group {f1.name!r} and {f2.name!r} keys differ:\n"
                             f"{keys1!r}\n{keys2!r}")

    iterator1 = ((k, f1[k], f2[k]) for k in recursive_keys(f2) if k not in skip)
    for k1, dset1, dset2 in iterator1:
        ar1, ar2 = dset1[:], dset2[:]
        _compare_arrays(ar1, ar2, err_msg=f'dataset {k1!r}\n')

        # Compare attributes
        if not dset1.attrs.keys() == dset2.attrs.keys():
            raise AssertionError(f"dataset {k1!r} attributes differ:\n"
                                 f"{list(dset1.attrs.keys())!r}\n{list(dset2.attrs.keys())!r}")
        iterator2 = ((k2, dset1.attrs[k2], dset1.attrs[k2]) for k2 in dset1.attrs.keys())
        for k2, attr1, attr2 in iterator2:
            _compare_arrays(attr1, attr2, err_msg=f'dataset {k1!r}; attribute {k2!r}')


def _compare_arrays(ar1: np.ndarray, ar2: np.ndarray, err_msg: str) -> None:
    __tracebackhide__ = True
    if issubclass(ar1.dtype.type, np.inexact):
        np.testing.assert_allclose(ar1, ar2, err_msg=err_msg)
    else:
        np.testing.assert_array_equal(ar1, ar2, err_msg=err_msg)
