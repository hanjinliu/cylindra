from typing import Callable, Generic, TypeVar

from dask import array as da
from dask import delayed as dask_delayed

_T = TypeVar("_T")


class Delayed(Generic[_T]):
    def compute(self) -> _T:
        ...


def compute(*args: Delayed[_T]) -> list[_T]:
    return list(da.compute(*args))


def delayed(func: Callable[..., _T]) -> Callable[..., Delayed[_T]]:
    return dask_delayed(func)
