from __future__ import annotations

import io
from abc import ABC, abstractmethod, abstractclassmethod
import json
from pathlib import Path
from typing import Any
from typing_extensions import Self


class BaseComponent(ABC):
    """Base class for all tomographic components."""

    @abstractclassmethod
    def from_dict(cls, js: dict[str, Any]) -> Self:
        """Construct a component from a dictionary."""
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert a component to a dictionary."""
        raise NotImplementedError

    def to_json(self, file_path: str | Path | io.IOBase) -> None:
        """
        Save spline model in a json format.

        Parameters
        ----------
        file_path : str
            Path to the file.
        """
        if isinstance(file_path, io.IOBase):
            return self._dump(file_path)
        with open(str(file_path).strip("'").strip('"'), mode="w") as f:
            self._dump(f)
        return None

    def _dump(self, f: io.IOBase) -> None:
        """Dump the project to a file."""
        return json.dump(self.to_dict(), f, indent=4, separators=(",", ": "))

    @classmethod
    def from_json(cls, file_path: str | Path | io.IOBase) -> Self:
        """
        Construct a spline model from a json file.

        Parameters
        ----------
        file_path : str
            Path to json file.

        Returns
        -------
        Spline
            Spline object constructed from the json file.
        """
        if isinstance(file_path, io.IOBase):
            return cls.from_dict(json.load(file_path))
        with open(str(file_path).strip("'").strip('"')) as f:
            js = json.load(f)
        return cls.from_dict(js)
