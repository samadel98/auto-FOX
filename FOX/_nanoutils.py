"""Utilities taken from `Nano-Utils <https://github.com/nlesc-nano/Nano-Utils>`_ 2.3.5.

Auto-FOX used to depend on Nano-Utils for the functions and classes below.
They are copied here, with shortened docstrings and without unused options,
so that Auto-FOX no longer needs it.
Nano-Utils is licensed under the Apache License 2.0 (see ``LICENSE-Apache-2.0.txt``),
and its NOTICE reads: "This product includes Nano-Utils, software developed by B. F. van Beek".

"""

from __future__ import annotations

import os
import sys
import shutil
import warnings
import importlib
from abc import ABCMeta, abstractmethod
from codecs import decode, encode
from collections import abc
from collections.abc import Callable, Generator, Hashable, Iterable, Mapping, MutableMapping
from contextlib import nullcontext
from functools import partial, wraps
from types import MappingProxyType
from typing import Any, IO, Generic, Literal, TypeVar, Union, cast, final

import h5py
import numpy as np
import yaml
from numpy.typing import ArrayLike, DTypeLike

__all__ = [
    "PathType",
    "EMPTY_MAPPING",
    "group_by_values",
    "raise_if",
    "set_docstring",
    "split_dict",
    "warning_filter",
    "get_importable",
    "as_nd_array",
    "fill_diagonal_blocks",
    "recursive_keys",
    "recursive_items",
    "delete_finally",
    "UniqueLoader",
    "Default",
    "Formatter",
    "supports_int",
    "supports_float",
    "isinstance_factory",
    "issubclass_factory",
    "import_factory",
    "AbstractFileContainer",
]

_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")
_FT = TypeVar("_FT", bound=Callable[..., Any])
_ST = TypeVar("_ST", bound="AbstractFileContainer")

#: An annotation for path-like objects.
PathType = Union[str, bytes, os.PathLike]

#: An immutable empty mapping.
EMPTY_MAPPING: Mapping[Any, Any] = MappingProxyType({})


def _null_func(obj: _T) -> _T:
    """Return the passed object."""
    return obj


def group_by_values(iterable: Iterable[tuple[_VT, _KT]]) -> dict[_KT, list[_VT]]:
    """Take an iterable of 2-tuples and group all first elements by the second."""
    ret: dict[_KT, list[_VT]] = {}
    for value, key in iterable:
        try:
            ret[key].append(value)
        except KeyError:
            ret[key] = [value]
    return ret


def raise_if(exception: None | BaseException) -> Callable[[_FT], _FT]:
    """Return a decorator that raises **exception** when the decorated function is called.

    The decorated function is returned unchanged if **exception** is :data:`None`.

    """
    if exception is None:
        return _null_func
    elif not isinstance(exception, BaseException):
        raise TypeError(f"'exception' expected None or an exception; "
                        f"observed type: {exception.__class__.__name__!r}")

    def decorator(func: _FT) -> _FT:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            raise exception
        return cast(_FT, wrapper)
    return decorator


def set_docstring(docstring: None | str) -> Callable[[_FT], _FT]:
    """Return a decorator that assigns **docstring** to the decorated function."""
    def decorator(func: _FT) -> _FT:
        func.__doc__ = docstring
        return func
    return decorator


def split_dict(
    dct: MutableMapping[_KT, _VT],
    preserve_order: bool = False,
    *,
    keep_keys: None | Iterable[_KT] = None,
    disgard_keys: None | Iterable[_KT] = None,
) -> dict[_KT, _VT]:
    """Pop all items from **dct** that are not in **keep_keys** (or are in **disgard_keys**).

    The popped items are returned as a new dictionary, always in the order of **dct**;
    **preserve_order** is only kept for compatibility with Nano-Utils.

    """
    if keep_keys is None and disgard_keys is None:
        raise TypeError("'keep_keys' and 'disgard_keys' cannot both be unspecified")
    elif keep_keys is not None and disgard_keys is not None:
        raise TypeError("'keep_keys' and 'disgard_keys' cannot both be specified")

    if keep_keys is not None:
        keep_set = set(keep_keys)
        keys = [k for k in dct if k not in keep_set]
    else:
        disgard_set = set(cast(Iterable[_KT], disgard_keys))
        keys = [k for k in dct if k in disgard_set]
    return {k: dct.pop(k) for k in keys}


def warning_filter(
    action: Literal["default", "error", "ignore", "always", "module", "once"],
    message: str = "",
    category: type[Warning] = Warning,
    module: str = "",
    lineno: int = 0,
    append: bool = False,
) -> Callable[[_FT], _FT]:
    """Return a decorator that calls the function under :func:`warnings.filterwarnings`."""
    def decorator(func: _FT) -> _FT:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with warnings.catch_warnings():
                warnings.filterwarnings(action, message, category, module, lineno, append)
                return func(*args, **kwargs)
        return cast(_FT, wrapper)
    return decorator


def get_importable(string: str, validate: None | Callable[[Any], bool] = None) -> Any:
    """Import the object named by a dotted **string**, *e.g.* ``"numpy.add"``.

    If **validate** is given, raise a :exc:`RuntimeError` if it does not return :data:`True`
    for the imported object.

    """
    try:
        head, *tail = string.split(".")
    except (AttributeError, TypeError) as ex:
        raise TypeError(f"'string' expected a str; observed type: "
                        f"{string.__class__.__name__!r}") from ex

    ret = importlib.import_module(head)
    for name in tail:
        ret = getattr(ret, name)

    if validate is not None and not validate(ret):
        name = getattr(validate, "__qualname__", repr(validate))
        raise RuntimeError(f"Passing {ret!r} to {name}() failed to return True")
    return ret


def as_nd_array(
    array: Iterable[Any] | ArrayLike,
    dtype: DTypeLike,
    ndmin: int = 1,
    copy: bool = False,
) -> np.ndarray:
    """Construct a numpy array from an iterable or array-like object.

    Unlike Nano-Utils, ``copy=False`` copies when needed rather than raising on NumPy >= 2.

    """
    try:
        return np.array(array, dtype=dtype, ndmin=ndmin, copy=True if copy else None)
    except TypeError:
        if not isinstance(array, abc.Iterable):
            raise
        ret: np.ndarray = np.fromiter(array, dtype=dtype)
        return ret.reshape(ret.shape + (ndmin - ret.ndim) * (1,))


def fill_diagonal_blocks(array: np.ndarray, i: int, j: int, val: float = np.nan) -> None:
    """Fill the diagonal blocks of size :math:`(i, j)` in the last two axes of **array**."""
    if j <= 0 or i <= 0:
        raise ValueError(f"'i' and 'j' should be larger than 0; observed values: {i} & {j}")

    i0 = j0 = 0
    dim1 = array.shape[-2]
    while dim1 > i0:
        array[..., i0:i0 + i, j0:j0 + j] = val
        i0 += i
        j0 += j


def _iter_dfs(group: h5py.Group) -> Generator[tuple[str, h5py.Dataset], None, None]:
    """Iterate depth-first through all datasets in **group**."""
    for value in group.values():
        if isinstance(value, h5py.Group):
            yield from _iter_dfs(value)
        else:
            yield value.name, value


def recursive_keys(group: h5py.Group) -> Generator[str, None, None]:
    """Yield the names of all datasets in **group** and its sub-groups."""
    for name, _ in _iter_dfs(group):
        yield name


def recursive_items(group: h5py.Group) -> Generator[tuple[str, h5py.Dataset], None, None]:
    """Yield the names and datasets of all datasets in **group** and its sub-groups."""
    yield from _iter_dfs(group)


def _delete_path(path: PathType, warn: bool = True) -> None:
    """Delete **path**, be it a file or a directory."""
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
        elif warn:
            warnings.warn(f"No such file or directory: {path!r}", RuntimeWarning, stacklevel=3)
    except Exception as ex:
        warning = RuntimeWarning(str(ex))
        warning.__cause__ = ex
        warnings.warn(warning, stacklevel=3)


def delete_finally(*paths: PathType, warn: bool = True) -> Callable[[_FT], _FT]:
    """Return a decorator that deletes **paths** after calling the decorated function."""
    def decorator(func: _FT) -> _FT:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            finally:
                for path in paths:
                    _delete_path(path, warn)
        return cast(_FT, wrapper)
    return decorator


class UniqueLoader(yaml.Loader):
    """A :class:`yaml.Loader` subclass that disallows duplicate keys."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        """Convert **node** into a :class:`dict`, raising for duplicate keys."""
        if not isinstance(node, yaml.MappingNode):
            raise yaml.constructor.ConstructorError(
                None, None, f"expected a mapping node, but found {node.id}", node.start_mark,
            )

        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    "found unhashable key", key_node.start_mark,
                )
            elif key in mapping:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    "found a duplicate key", key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


class Default(Generic[_T]):
    """A :mod:`schema` validator that returns a default value, calling it if callable."""

    def __init__(self, value: _T, call: bool = True) -> None:
        """Initialize an instance."""
        self.value = value
        self.call = call

    def __repr__(self) -> str:
        """Implement :func:`repr(self) <repr>`."""
        return f"{self.__class__.__name__}({self.value!r}, call={self.call!r})"

    def validate(self, *args: object, **kwargs: object) -> Any:
        """Return the default value, ignoring the passed data."""
        if self.call and callable(self.value):
            return self.value()
        return self.value


class Formatter(str):
    """A :class:`str` subclass for :mod:`schema` error messages.

    The message can contain the ``{name}``, ``{value}`` and ``{type}`` fields.

    """

    def __init__(self, msg: str) -> None:
        """Initialize an instance."""
        self._msg = msg

    def __repr__(self) -> str:
        """Implement :func:`repr(self) <repr>`."""
        return f"{self.__class__.__name__}(msg={self._msg!r})"

    def format(self, obj: object) -> str:  # type: ignore[override]
        """Return the formatted message for **obj**."""
        try:
            name = self._msg.split("'", maxsplit=2)[1]
        except IndexError:
            name = ""
        try:
            return self._msg.format(name=name or "value", value=obj, type=obj.__class__.__name__)
        except Exception as ex:
            warning = RuntimeWarning(ex)
            warning.__cause__ = ex
            warnings.warn(warning)
            return repr(obj)

    @property
    def __mod__(self) -> Callable[[object], str]:  # type: ignore[override]
        """Get :meth:`Formatter.format`; :mod:`schema` formats its messages with ``%``."""
        return self.format


def supports_int(value: object) -> bool:
    """Check if **value** can be converted into an integer without loss of precision."""
    try:
        int(value)  # type: ignore[call-overload]
        return float(value).is_integer()  # type: ignore[arg-type]
    except Exception:
        return False


def supports_float(value: object) -> bool:
    """Check if **value** can be converted into a float."""
    try:
        float(value)  # type: ignore[arg-type]
        return True
    except Exception:
        return False


def isinstance_factory(class_or_tuple: type | tuple[type, ...]) -> Callable[[object], bool]:
    """Return a function that checks if an object is an instance of **class_or_tuple**."""
    def func(obj: object) -> bool:
        return isinstance(obj, class_or_tuple)
    func.__doc__ = f"Return :code:`isinstance(obj, {class_or_tuple!r})`."
    return func


def issubclass_factory(class_or_tuple: type | tuple[type, ...]) -> Callable[[type], bool]:
    """Return a function that checks if a class is a subclass of **class_or_tuple**."""
    def func(cls: type) -> bool:
        return issubclass(cls, class_or_tuple)
    func.__doc__ = f"Return :code:`issubclass(cls, {class_or_tuple!r})`."
    return func


def import_factory(validate: Callable[[Any], bool]) -> Callable[[str], Any]:
    """Return a function that imports an object with :func:`get_importable` and **validate**."""
    if not callable(validate):
        raise TypeError(f"'validate' expected a callable; observed type: "
                        f"{validate.__class__.__name__!r}")
    return partial(get_importable, validate=validate)


def _file_to_context(file: PathType | IO[Any], **kwargs: Any) -> Any:
    """Open a path-like object, or wrap a file-like object in a no-op context manager."""
    try:
        return open(file, **kwargs)  # type: ignore[arg-type]
    except TypeError:
        return nullcontext(file)


class AbstractFileContainer(metaclass=ABCMeta):
    """An abstract container for reading and writing files.

    Subclasses define how a file is parsed in :meth:`_read` and how it is written
    in :meth:`_write`.

    """

    __slots__ = ()

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize an instance."""
        raise NotImplementedError("Trying to call an abstract method")

    @final
    @classmethod
    def read(
        cls: type[_ST],
        file: PathType | IO[Any],
        bytes_decoding: None | str = None,
        **kwargs: Any,
    ) -> _ST:
        r"""Construct a new instance from this object's class by reading the content of **file**.

        Parameters
        ----------
        file : :class:`str`, :class:`bytes`, :class:`os.PathLike` or :class:`~typing.IO`
            A path- or file-like object.
        bytes_decoding : :class:`str`, optional
            The encoding to use when **file** is (or will be) opened in :class:`bytes` mode.
            This value should be left empty otherwise.
        \**kwargs : :data:`~typing.Any`
            Further keyword arguments for :func:`open`.
            Only relevant if **file** is a path-like object.

        Returns
        -------
        :class:`AbstractFileContainer`
            A new instance constructed from **file**.

        """
        kwargs.setdefault("mode", "r")
        with _file_to_context(file, **kwargs) as f:
            try:
                assert f.readable()
            except (TypeError, AttributeError) as ex:
                raise TypeError("'file' expected a file- or path-like object; "
                                f"observed type: {file.__class__.__name__!r}") from ex
            except AssertionError:
                f.read()  # This will raise an `io.UnsupportedOperation`

            if bytes_decoding is None:
                decoder: Callable[[Any], str] = _null_func
            else:
                decoder = partial(decode, encoding=bytes_decoding)  # type: ignore[assignment]
            cls_dict = cls._read(f, decoder)

        ret = cls(**cls_dict)
        ret._read_postprocess()
        return ret

    @classmethod
    @abstractmethod
    def _read(cls, file_obj: IO[Any], decoder: Callable[[Any], str]) -> dict[str, Any]:
        """A helper function for :meth:`~AbstractFileContainer.read`.

        Parameters
        ----------
        file_obj : :class:`IO[AnyStr]<typing.IO>`
            A file-like object opened in read mode.
        decoder : :data:`Callable[[AnyStr], str]<typing.Callable>`
            A function for converting the items of **file_obj** into strings.

        Returns
        -------
        :class:`dict[str, Any]<dict>`
            A dictionary with keyword arguments for a new instance of this objects' class.

        """
        raise NotImplementedError("Trying to call an abstract method")

    def _read_postprocess(self) -> None:
        """Post process new instances created by :meth:`~AbstractFileContainer.read`."""
        pass

    @final
    def write(
        self,
        file: PathType | IO[Any] = sys.stdout,
        bytes_encoding: None | str = None,
        **kwargs: Any,
    ) -> None:
        r"""Write the content of this instance to **file**.

        Parameters
        ----------
        file : :class:`str`, :class:`bytes`, :class:`os.PathLike` or :class:`~typing.IO`
            A path- or file-like object.
            Defaults to :data:`sys.stdout` if not specified.
        bytes_encoding : :class:`str`, optional
            The encoding to use when **file** is (or will be) opened in :class:`bytes` mode.
            This value should be left empty otherwise.
        \**kwargs : :data:`~typing.Any`
            Further keyword arguments for :func:`open`.
            Only relevant if **file** is a path-like object.

        """
        kwargs.setdefault("mode", "w")
        with _file_to_context(file, **kwargs) as f:
            try:
                assert f.writable()
            except (TypeError, AttributeError) as ex:
                raise TypeError("'file' expected a file- or path-like object; "
                                f"observed type: {file.__class__.__name__!r}") from ex
            except AssertionError:
                f.write(None)  # This will raise an `io.UnsupportedOperation`

            if bytes_encoding is None:
                encoder: Callable[[str], str | bytes] = _null_func
            else:
                encoder = partial(encode, encoding=bytes_encoding)
            self._write(f, encoder)

    @abstractmethod
    def _write(self, file_obj: IO[Any], encoder: Callable[[str], Any]) -> None:
        """A helper function for :meth:`~AbstractFileContainer.write`.

        Parameters
        ----------
        file_obj : :class:`IO[AnyStr]<typing.IO>`
            A file-like object opened in write mode.
        encoder : :data:`Callable[[str], AnyStr]<typing.Callable>`
            A function for converting strings into either :class:`str` or :class:`bytes`,
            the exact type matching that of **file_obj**.

        """
        raise NotImplementedError("Trying to call an abstract method")
