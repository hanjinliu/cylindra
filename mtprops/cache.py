from __future__ import annotations
from typing import TypeVar
from collections import OrderedDict

_V = TypeVar("_V")

class CacheMap:
    def __init__(self, maxgb:float=2.0):
        self.maxgb = maxgb
        self.cache: OrderedDict[int, _V] = OrderedDict()
        self.gb = 0.0
    
    def __getitem__(self, key: tuple[str, int]) -> _V:
        real_key, identifier = key
        idf, value = self.cache[real_key]
        if idf == identifier:
            return value
        else:
            raise KeyError("Wrong identifier")
    
    def __setitem__(self, key: tuple[str, int], value:_V):
        real_key, identifier = key
        self.cache[real_key] = (identifier, value)
        size = sum(a.nbytes for a in value)/1e9
        self.gb += size
        while self.gb > self.maxgb:
            self.pop()
    
    def pop(self) -> None:
        _, item = self.cache.popitem(last=False)
        self.gb -= sum(a.nbytes for a in item[1])/1e9
        return None

    def keys(self):
        return self.cache.keys()
    
    def clear(self):
        self.cache.clear()
        self.gb = 0
        return None
