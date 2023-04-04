from __future__ import annotations

from abc import ABC, abstractmethod, abstractclassmethod
import json
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
    
    def to_json(self, file_path: str) -> None:
        """
        Save spline model in a json format.

        Parameters
        ----------
        file_path : str
            Path to the file.
        """
        file_path = str(file_path)
        
        with open(file_path, mode="w") as f:
            json.dump(self.to_dict(), f, indent=4, separators=(", ", ": "))
        
        return None
    
    @classmethod
    def from_json(cls, file_path: str) -> Self:
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
        file_path = str(file_path)
        
        with open(file_path, mode="r") as f:
            js = json.load(f)
        return cls.from_dict(js)
