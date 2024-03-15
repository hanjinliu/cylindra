from __future__ import annotations

import re
from pathlib import Path
from typing import overload

from acryo import BatchLoader
from psygnal.containers import EventedList

from ._utils import LoaderInfo


class LoaderList(EventedList[LoaderInfo]):
    @overload
    def __getitem__(self, index: int | str) -> LoaderInfo:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[LoaderInfo]:
        ...

    def __getitem__(self, index) -> LoaderInfo:
        if isinstance(index, str):
            index = self.find(index)
        return super().__getitem__(index)

    def __setitem__(self, index, value):
        if not isinstance(value, LoaderInfo):
            raise TypeError(f"Expected LoaderInfo, got {type(value)}")
        return super().__setitem__(index, value)

    def __delitem__(self, key: int | slice | str) -> None:
        if isinstance(key, str):
            key = self.find(key)
        return super().__delitem__(key)

    def find(self, name: str, default: int | None = None) -> int:
        for i, info in enumerate(self):
            if info.name == name:
                return i
        if default is None:
            raise ValueError(f"Loader {name!r} not found")
        return default

    def insert(self, index: int, value: LoaderInfo) -> None:
        if not isinstance(value, LoaderInfo):
            raise TypeError(f"Expected LoaderInfo, got {type(value)}")
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

    def add_loader(
        self,
        loader: BatchLoader,
        name: str,
        image_paths: dict[int, Path],
        invert: dict[int, bool] | None = None,
    ) -> LoaderInfo:
        """Add a new loader to the list."""
        if invert is None:
            invert = {k: False for k in image_paths.keys()}
        info = LoaderInfo(loader, name, image_paths, invert)
        self.append(info)
        return info
