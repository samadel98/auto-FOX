"""Tests for :mod:`FOX._nanoutils` and :mod:`FOX._dataclass`."""

from __future__ import annotations

import io
import os
import copy
import warnings
from typing import Any, IO, Callable
from pathlib import Path

import h5py
import yaml
import numpy as np
import pytest

from FOX._dataclass import AbstractDataClass
from FOX._nanoutils import (
    EMPTY_MAPPING,
    AbstractFileContainer,
    Default,
    Formatter,
    UniqueLoader,
    as_nd_array,
    delete_finally,
    fill_diagonal_blocks,
    get_importable,
    group_by_values,
    import_factory,
    isinstance_factory,
    issubclass_factory,
    raise_if,
    recursive_items,
    recursive_keys,
    set_docstring,
    split_dict,
    supports_float,
    supports_int,
    warning_filter,
)


def test_empty_mapping() -> None:
    assert len(EMPTY_MAPPING) == 0
    with pytest.raises(TypeError):
        EMPTY_MAPPING["a"] = 1  # type: ignore[index]


def test_group_by_values() -> None:
    assert group_by_values([(1, "a"), (2, "b"), (3, "a")]) == {"a": [1, 3], "b": [2]}


def test_raise_if() -> None:
    def func() -> int:
        return 1

    assert raise_if(None)(func) is func
    with pytest.raises(ImportError, match="test"):
        raise_if(ImportError("test"))(func)()
    with pytest.raises(TypeError):
        raise_if(1)  # type: ignore[arg-type]


def test_set_docstring() -> None:
    @set_docstring("test")
    def func() -> None:
        pass

    assert func.__doc__ == "test"


@pytest.mark.parametrize("preserve_order", [False, True])
def test_split_dict(preserve_order: bool) -> None:
    dct = {1: "a", 2: "b", 3: "c", 4: "d"}
    assert split_dict(dct, preserve_order, keep_keys={1, 2}) == {3: "c", 4: "d"}
    assert dct == {1: "a", 2: "b"}
    assert split_dict(dct, preserve_order, disgard_keys=[1]) == {1: "a"}
    assert dct == {2: "b"}
    with pytest.raises(TypeError):
        split_dict(dct)
    with pytest.raises(TypeError):
        split_dict(dct, keep_keys=[1], disgard_keys=[2])


def test_warning_filter() -> None:
    @warning_filter("error", category=UserWarning)
    def func() -> None:
        warnings.warn("test", UserWarning)

    with pytest.raises(UserWarning, match="test"):
        func()
    with pytest.warns(UserWarning, match="test"):
        warnings.warn("test", UserWarning)  # The filter is restored afterwards


def test_get_importable() -> None:
    assert get_importable("numpy.add") is np.add
    assert get_importable("numpy.add", validate=callable) is np.add
    with pytest.raises(RuntimeError):
        get_importable("numpy.pi", validate=callable)
    with pytest.raises(TypeError):
        get_importable(1)  # type: ignore[arg-type]


def test_as_nd_array() -> None:
    ar = np.arange(3.0)
    assert as_nd_array(ar, float) is ar
    assert as_nd_array(ar, float, copy=True) is not ar
    np.testing.assert_array_equal(as_nd_array([1, 2], int, ndmin=2), [[1, 2]])
    np.testing.assert_array_equal(as_nd_array(iter([1, 2]), float), [1.0, 2.0])
    assert as_nd_array(iter([1, 2]), float, ndmin=2).shape == (2, 1)
    with pytest.raises(TypeError):
        as_nd_array(1, object(), copy=True)  # type: ignore[call-overload]


def test_fill_diagonal_blocks() -> None:
    ar = np.zeros((4, 6), dtype=int)
    fill_diagonal_blocks(ar, 2, 3, val=1)
    ref = np.zeros((4, 6), dtype=int)
    ref[:2, :3] = ref[2:, 3:] = 1
    np.testing.assert_array_equal(ar, ref)
    with pytest.raises(ValueError):
        fill_diagonal_blocks(ar, 0, 3)


def test_recursive_keys(tmp_path: Path) -> None:
    with h5py.File(tmp_path / "test.hdf5", "w") as f:
        f.create_dataset("a", data=[1])
        f.create_dataset("b/c", data=[2])
        f.create_dataset("b/d/e", data=[3])
        assert list(recursive_keys(f)) == ["/a", "/b/c", "/b/d/e"]
        assert [(k, v[0]) for k, v in recursive_items(f)] == [("/a", 1), ("/b/c", 2), ("/b/d/e", 3)]


def test_delete_finally(tmp_path: Path) -> None:
    file = tmp_path / "file.txt"
    directory = tmp_path / "dir"

    @delete_finally(file, directory)
    def func() -> None:
        file.write_text("test")
        directory.mkdir()
        raise RuntimeError

    with pytest.raises(RuntimeError):
        func()
    assert not file.exists() and not directory.exists()

    with pytest.warns(RuntimeWarning):
        delete_finally(file)(lambda: None)()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        delete_finally(os.fspath(file), warn=False)(lambda: None)()


def test_unique_loader() -> None:
    assert yaml.load("a: 0\nb: 1", Loader=UniqueLoader) == {"a": 0, "b": 1}
    with pytest.raises(yaml.constructor.ConstructorError, match="found a duplicate key"):
        yaml.load("a: 0\na: 1", Loader=UniqueLoader)
    with pytest.raises(yaml.constructor.ConstructorError, match="found unhashable key"):
        yaml.load("? [a]\n: 1", Loader=UniqueLoader)


def test_schema_helpers() -> None:
    default = Default(list)
    assert default.validate(None) == []
    assert Default(list, call=False).validate(None) is list
    assert repr(default) == "Default(<class 'list'>, call=True)"

    fmt = Formatter("'a' expected an int; observed type: {type!r}, value: {value!r}, {name}")
    assert repr(fmt) == f"Formatter(msg={fmt._msg!r})"
    assert fmt.format(1.5) == "'a' expected an int; observed type: 'float', value: 1.5, a"
    with pytest.warns(RuntimeWarning):
        assert Formatter("{bob}").format(1) == "1"

    assert fmt % "b" == fmt.format("b")  # schema formats its error messages with `%`

    assert supports_int(1.0) and supports_int("1") and not supports_int(1.5)
    assert supports_float("1.5") and not supports_float("a")

    assert isinstance_factory(int)(1) and not isinstance_factory(int)(1.0)
    assert issubclass_factory(int)(bool) and not issubclass_factory(int)(float)
    assert import_factory(callable)("numpy.add") is np.add
    with pytest.raises(TypeError):
        import_factory(1)  # type: ignore[arg-type]


class _Container(AbstractFileContainer):
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    @classmethod
    def _read(cls, file_obj: IO[Any], decoder: Callable[[Any], str]) -> dict[str, Any]:
        return {"lines": [decoder(i).rstrip("\n") for i in file_obj]}

    def _write(self, file_obj: IO[Any], encoder: Callable[[str], Any]) -> None:
        for i in self.lines:
            file_obj.write(encoder(f"{i}\n"))


def test_file_container(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        AbstractFileContainer()  # type: ignore[abstract]

    obj = _Container(["a", "b"])
    obj.write(tmp_path / "test.txt")
    assert _Container.read(tmp_path / "test.txt").lines == ["a", "b"]

    obj.write(tmp_path / "test.bin", bytes_encoding="utf-8", mode="wb")
    with open(tmp_path / "test.bin", "rb") as f:
        assert _Container.read(f, bytes_decoding="utf-8").lines == ["a", "b"]

    buffer = io.StringIO()
    obj.write(buffer)
    assert buffer.getvalue() == "a\nb\n"
    with pytest.raises(TypeError):
        _Container.read(1.0)  # type: ignore[arg-type]
    with pytest.raises(io.UnsupportedOperation):
        _Container.read(tmp_path / "test.txt", mode="a")


class _DataClass(AbstractDataClass):
    _PRIVATE_ATTR = frozenset({"c"})

    def __init__(self, a: int, b: list[int]) -> None:
        super().__init__()
        self.a = a
        self.b = b
        self.c = None

    @AbstractDataClass.inherit_annotations()
    def __copy__(self):
        return self.copy(deep=False)


def test_dataclass() -> None:
    obj = _DataClass(1, [2])
    assert repr(obj) == "_DataClass(\n    a = 1,\n    b = [2]\n)"
    assert obj == _DataClass(1, [2]) and obj != _DataClass(1, [3])
    assert obj.as_dict() == {"a": 1, "b": [2]}
    assert _DataClass.from_dict(obj.as_dict()) == obj
    assert copy.copy(obj).b is obj.b
    assert copy.deepcopy(obj).b is not obj.b
    assert "_DataClass.copy" in _DataClass.__copy__.__doc__
    assert hash(obj) == hash(obj)  # `b` is a list, so its id() is used for the hash
