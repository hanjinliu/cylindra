from contextlib import contextmanager
from functools import wraps
from typing import Any

import pytest


class pytest_group:
    """
    Class to group tests and skip them if one fails many times

    Examples
    --------
    >>> @pytest_group("test_group", maxfail=1)
    >>> def test_something_1():
    >>>     ...

    >>> @pytest_group("test_group", maxfail=1)
    >>> def test_something_2():
    >>>     ...

    """

    _all = {}

    def __new__(cls, id: str, maxfail: int = 1):  # noqa: ARG003
        if self := cls._all.get(id):
            return self
        cls._all[id] = self = super().__new__(cls)
        return self

    def __init__(self, id: str, maxfail: int = 1) -> None:
        self._fail_count = 0
        self._maxfail = maxfail
        self._id = id

    def __call__(self, f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if self._fail_count >= self._maxfail:
                return pytest.skip(f"Previous test failed: {self._id}")
            try:
                return f(*args, **kwargs)
            except Exception as e:
                self._fail_count += 1
                raise e

        return wrapper


class ExceptionGroup:
    """Custom exception group to test many cases without launching the viewer many times."""

    def __init__(self, max_fail: int = 9999) -> None:
        self._list = list[tuple[Exception, Any]]()
        self._max_fail = max_fail

    @property
    def nfail(self) -> int:
        return len(self._list)

    @contextmanager
    def merging(self, desc=None):
        try:
            yield
        except Exception as e:
            self._list.append((e, desc))
        if self.nfail >= self._max_fail:
            self.raise_exceptions()

    def raise_exceptions(self) -> None:
        if self._list:
            lines = "\n\t".join(f"({desc!r}) {exc}" for exc, desc in self._list)
            ori_exc = self._list[-1][0]
            raise RuntimeError(f"{self.nfail} test failed:\n\t{lines}") from ori_exc
