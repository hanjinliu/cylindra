from functools import wraps
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

    def __new__(cls, id: str, maxfail: int = 1):
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
