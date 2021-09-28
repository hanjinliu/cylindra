from __future__ import annotations
from typing import TypeVar
from collections import OrderedDict
import numpy as np

_K = TypeVar("_K")
_V = TypeVar("_V")

class AbstractCacheMap:
    def __init__(self, maxgb:float=2.0):
        self.maxgb = maxgb
        self.cache: OrderedDict[_K, _V] = OrderedDict()
        self.gb = 0.0
    
    def __getitem__(self, key: _K) -> _V:
        return self.cache[key]
        
    def __setitem__(self, key: _K, value:_V):
        self.cache[key] = value
        size = self.getsize(value)
        self.gb += size
        while self.gb > self.maxgb:
            self.pop()
    
    def pop(self) -> None:
        _, value = self.cache.popitem(last=False)
        self.gb -= self.getsize(value)
        return None

    def keys(self):
        return self.cache.keys()
    
    def clear(self):
        self.cache.clear()
        self.gb = 0
        return None

    def getsize(self, value: _V):
        raise NotImplementedError()


class ArrayCacheMap(AbstractCacheMap):
    def getsize(self, value: np.ndarray):
        return value.nbytes/1e9