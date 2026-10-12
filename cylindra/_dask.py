from typing import Callable

from dask import array as da
from dask import delayed as dask_delayed


class Delayed[T]:
    def compute(self) -> T: ...


def compute[T](*args: Delayed[T]) -> list[T]:
    return list(da.compute(*args))


def delayed[T](func: Callable[..., T]) -> Callable[..., Delayed[T]]:
    return dask_delayed(func)
