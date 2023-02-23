from __future__ import annotations

from typing import overload
import dataclasses
import re
from pathlib import Path
import weakref
from psygnal.containers import EventedList

from acryo import BatchLoader

@dataclasses.dataclass(frozen=True)
class LoaderInfo:
    loader: BatchLoader
    name: str
    paths: list[tuple[Path, list[Path]]] = dataclasses.field(default_factory=list)
    predicate: "str | None" = None
    parent: "weakref.ReferenceType[LoaderInfo] | None" = None
    
    def get_parent(self):
        if self.parent is None:
            return None
        return self.parent()
    
    def rename(self, name: str):
        return LoaderInfo(self.loader, name, self.paths, self.predicate, self.parent)

class LoaderList(EventedList[LoaderInfo]):
    @overload
    def __getitem__(self, index: int | str) -> LoaderInfo: ...
    @overload
    def __getitem__(self, index: slice) -> list[LoaderInfo]: ...
    
    def __getitem__(self, index) -> LoaderInfo:
        if isinstance(index, str):
            index = self.find(index)
        return super().__getitem__(index)
    
    def __setitem__(self, index, value):
        if not isinstance(value, LoaderInfo):
            raise TypeError(f"Expected LoaderInfo, got {type(value)}")
        return super().__setitem__(index, value)
    
    def __delitem__(self, key: int | slice | str) -> None:
        if isinstance(index, str):
            index = self.find(index)
        return super().__delitem__(key)
    
    def find(self, name: str, default: int | None = None) -> int:
        for i, info in enumerate(self): 
            if info.name == name:
                return i
        if default is None:
            raise ValueError(f"Loader {name!r} not found")
        return -1
    
    def insert(self, index: int, value: LoaderInfo) -> None:
        name = value.name
        if self.find(name, default=-1) >= 0:
            if re.match(r".+-\d+$", name):
                prefix, s0 = name.rsplit("-", 1)
                suffix = int(s0)
            else:
                prefix, suffix = name, 0
            while self.find(f"{prefix}-{suffix}", default=-1) > 0:
                suffix += 1
            value = value.rename(f"{prefix}-{suffix}")
        return super().insert(index, value)

